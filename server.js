#!/usr/bin/env node
/*
  SwitchAI Cerebras Proxy Server
  --------------------------------
  Endpoints:
    GET  /health                  -> { status: 'ok', ts }
    POST /cerebras/chat           -> Non-stream JSON completion
      Body: { model, messages, temperature?, top_p?, max_tokens?, apiKey? }
    POST /cerebras/chat/stream    -> Server-Sent Events streaming
      Body: { model, messages, temperature?, top_p?, max_tokens?, apiKey? }
    POST /mistral/chat            -> Non-stream JSON completion (Mistral)
      Body: { model, messages, temperature?, top_p?, max_tokens?, apiKey? }
    POST /mistral/chat/stream     -> Server-Sent Events streaming (Mistral)
    POST /gemini/chat             -> Non-stream JSON completion (Gemini)
      Body: { model, messages, temperature?, top_p?, max_tokens?, apiKey? }
    POST /gemini/chat/stream      -> Server-Sent Events streaming (Gemini)
      Body: { model, messages, temperature?, top_p?, max_tokens?, apiKey? }

  apiKey precedence: request body apiKey > process.env.{PROVIDER}_API_KEY

  Example non-stream request:
    curl -X POST http://localhost:5058/cerebras/chat -H 'Content-Type: application/json' \
      -d '{"model":"qwen-3-coder-480b","messages":[{"role":"user","content":"Hello"}]}'

  Example stream request:
    curl -N -X POST http://localhost:5058/cerebras/chat/stream -H 'Content-Type: application/json' \
      -d '{"model":"qwen-3-coder-480b","messages":[{"role":"user","content":"Hello"}]}'
*/
import Cerebras from '@cerebras/cerebras_cloud_sdk';
import { Mistral } from '@mistralai/mistralai';
import { GoogleGenAI } from '@google/genai';
import compression from 'compression';
import cors from 'cors';
import crypto from 'crypto';
import 'dotenv/config';
import express from 'express';
import monitoring from './serverMonitoring.js';

const app = express();

// Performance optimizations
app.use(compression()); // Enable gzip compression
app.use(cors({ origin: '*', maxAge: 600 }));
app.use(express.json({ limit: '2mb' })); // Use built-in JSON parser
app.disable('x-powered-by'); // Remove Express header for security

// Logging -----------------------------------------------------------
const LOG_LEVEL = (process.env.LOG_LEVEL || 'info').toLowerCase();
const levels = ['error','warn','info','debug'];
function shouldLog(l){ return levels.indexOf(l) <= levels.indexOf(LOG_LEVEL); }
function log(l, msg, meta){ if(!shouldLog(l)) return; const ts=new Date().toISOString(); try{ console.log(JSON.stringify({ ts, level:l, msg, ...(meta||{}) })); } catch { console.log(ts,l,msg,meta||''); } }

// Request/response logging middleware
app.use((req,res,next)=>{
  const id = crypto.randomUUID();
  const start = process.hrtime.bigint();
  const ip = req.headers['x-forwarded-for'] || req.socket.remoteAddress;
  const method = req.method;
  const url = req.originalUrl || req.url;
  const reqSize = Number(req.headers['content-length']||0);
  log('info','req.begin',{ id, method, url, ip, reqSize });
  let finished = false;
  function done(){
    if(finished) return; finished = true;
    const durNs = Number(process.hrtime.bigint() - start);
const durMs = Number(durNs) / 1_000_000; // now durMs is a Number
    const status = res.statusCode;
    const respSize = res.getHeader('content-length') || null;
    log('info','req.end',{ id, method, url, status, durMs, respSize });
    
    // Record in monitoring
    monitoring.recordRequest({
      endpoint: url,
      statusCode: status,
      responseTime: durMs,
      error: status >= 400 ? res.statusMessage : null,
      method,
    });
  }
  res.on('finish', done); res.on('close', done); res.on('error', done);
  req._reqId = id; // attach for downstream use
  next();
});

function buildClient(apiKey) {
  if (!apiKey) throw new Error('Missing Cerebras API key');
  return new Cerebras({ apiKey });
}
function buildMistralClient(apiKey) {
  if (!apiKey) throw new Error('Missing Mistral API key');
  return new Mistral({ apiKey });
}

function buildGeminiClient(apiKey) {
  if (!apiKey) throw new Error('Missing Gemini API key');
  return new GoogleGenAI({ apiKey });
}

// Convert OpenAI messages to Gemini contents format (supports text + images)
function convertToGeminiContents(messages) {
  const normalized = normalizeMessages(messages);
  return normalized.map(msg => {
    const role = msg.role === 'assistant' ? 'model' : 'user';
    
    // Handle string content (text-only)
    if (typeof msg.content === 'string') {
      return { role, parts: [{ text: msg.content }] };
    }
    
    // Handle array content (multimodal: text + images)
    if (Array.isArray(msg.content)) {
      const parts = [];
      for (const part of msg.content) {
        if (typeof part === 'string') {
          parts.push({ text: part });
        } else if (part.type === 'text') {
          parts.push({ text: part.text || '' });
        } else if (part.type === 'image_url') {
          // Convert OpenAI image_url format to Gemini inlineData format
          const imageUrl = part.image_url?.url || part.image_url;
          if (imageUrl && imageUrl.startsWith('data:')) {
            // Extract base64 data and mime type from data URL
            const match = imageUrl.match(/^data:([^;]+);base64,(.+)$/);
            if (match) {
              parts.push({
                inlineData: {
                  mimeType: match[1],
                  data: match[2]
                }
              });
            }
          }
        }
      }
      return { role, parts: parts.length > 0 ? parts : [{ text: '' }] };
    }
    
    // Fallback
    return { role, parts: [{ text: String(msg.content || '') }] };
  });
}

function normalizeMessages(raw) {
  if (!Array.isArray(raw)) return [];
  return raw.map(m => {
    const role = m.role || 'user';
    // Support either string content or already structured array (for Mistral multimodal: [{type:'text',text:'..'},{type:'image_url',image_url:{url:'...'}}])
    if (Array.isArray(m.content)) {
      // Light validation: ensure objects with type
      const parts = m.content.filter(p => p && (typeof p === 'string' || typeof p.type === 'string'));
      return { role, content: parts };
    }
    return { role, content: m.content || '' };
  });
}

// Collapse structured content parts (text + images) into a text-only delta summary for streaming line-by-line if provider returns object chunks
function flattenPartsToText(content) {
  try {
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
      return content.map(part => {
        if (!part) return '';
        if (typeof part === 'string') return part;
        if (part.type === 'text' && typeof part.text === 'string') return part.text;
        if (part.type === 'image_url') return ''; // omit image placeholders from text surface
        return '';
      }).join('');
    }
    return '';
  } catch { return ''; }
}

// Transform OpenAI-style parts into Mistral expected shapes.
// Incoming examples:
//  { type:'text', text:'Hello' }
//  { type:'image_url', image_url:{ url:'https://...' } }
// Mistral vision expects either plain string (text only) OR array of blocks each with an accepted type.
// We'll convert image_url -> { type:'document_url', documentUrl:'<url>' }
function transformForMistralVision(messages) {
  return messages.map(m => {
    if (!Array.isArray(m.content)) return m; // simple string case already fine
    const transformed = m.content.map(part => {
      if (!part) return null;
      if (typeof part === 'string') return { type:'text', text: part };
      const t = String(part.type||'').toLowerCase();
      if (t === 'text' && typeof part.text === 'string') {
        return { type: 'text', text: part.text };
      }
      if (t === 'image_url' && part.image_url && typeof part.image_url.url === 'string') {
        return { type: 'document_url', documentUrl: part.image_url.url };
      }
      // Unsupported part types for now: silently drop
      return null;
    }).filter(Boolean);
    // If after transform we only have text parts, and exactly one, collapse back to string to satisfy strict schemas
    if (transformed.length === 1 && transformed[0].type === 'text') {
      return { ...m, content: transformed[0].text };
    }
    // If empty, fallback to empty string to avoid schema errors
    if (!transformed.length) return { ...m, content: '' };
    return { ...m, content: transformed };
  });
}

app.get('/health', (req, res) => {
  res.json({ status: 'ok', ts: Date.now() });
});

// Server status endpoint with comprehensive metrics
app.get('/api/status', (req, res) => {
  try {
    const status = monitoring.getStatus();
    res.json({
      status: 'ok',
      timestamp: Date.now(),
      ...status,
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: error.message,
      timestamp: Date.now(),
    });
  }
});

// Simple health check for monitoring
app.get('/api/health', (req, res) => {
  const health = monitoring.getHealth();
  const statusCode = health.status === 'healthy' ? 200 : health.status === 'critical' ? 503 : 200;
  res.status(statusCode).json(health);
});

const CEREBRAS_HARD_MAX = 32768;
function capTokens(v){
  if (typeof v !== 'number' || !isFinite(v) || v <= 0) return 4096;
  return Math.min(CEREBRAS_HARD_MAX, Math.max(1, Math.floor(v)));
}

app.post('/cerebras/chat', async (req, res) => {
  let { model, messages, temperature = 0.7, top_p = 0.8, max_tokens = 8192, apiKey } = req.body || {};
  max_tokens = capTokens(max_tokens);
  res.setHeader('x-request-id', req._reqId || '');
  try {
    if (!model) return res.status(400).json({ error: { message: 'model required', code: 'model_required' } });
    const key = apiKey || process.env.CEREBRAS_API_KEY;
    const client = buildClient(key);
    const payload = {
      model,
      messages: normalizeMessages(messages),
      stream: false,
      max_completion_tokens: max_tokens,
      temperature,
      top_p
    };
    const t0 = Date.now();
    log('debug','cerebras.request', { id: req._reqId, model, stream:false, temp:temperature, top_p, max_tokens });
    let data;
    try {
      data = await client.chat.completions.create(payload);
    } catch (err) {
      const msg = err?.message || String(err);
      const lower = msg.toLowerCase();
      const isModel404 = lower.includes('does not exist') || lower.includes('model_not_found');
      if (isModel404) {
        log('warn','cerebras.model_not_found', { id: req._reqId, model, msg });
        return res.status(404).json({ error: { message: msg, code: 'model_not_found' } });
      }
      log('error','cerebras.upstream_error', { id: req._reqId, model, msg });
      return res.status(502).json({ error: { message: msg, code: 'upstream_error' } });
    }
    const latency_ms = Date.now() - t0;
    log('info','cerebras.response', { id: req._reqId, model, latency_ms, usage: data?.usage });
    res.json({ ...data, latency_ms, request_id: req._reqId });
  } catch (e) {
    const msg = e?.message || String(e);
    log('error','cerebras.error', { id: req._reqId, error: msg });
    res.status(500).json({ error: { message: msg, code: 'internal_error' } });
  }
});

app.post('/cerebras/chat/stream', async (req, res) => {
  let { model, messages, temperature = 0.7, top_p = 0.8, max_tokens = 8192, apiKey } = req.body || {};
  max_tokens = capTokens(max_tokens);
  res.setHeader('x-request-id', req._reqId || '');
  if (!model) return res.status(400).json({ error: { message: 'model required', code: 'model_required' } });
  let stream;
  try {
    const key = apiKey || process.env.CEREBRAS_API_KEY;
    const client = buildClient(key);
    const payload = {
      model,
      messages: normalizeMessages(messages),
      stream: true,
      max_completion_tokens: max_tokens,
      temperature,
      top_p
    };
    log('debug','cerebras.stream.begin', { id: req._reqId, model, temp:temperature, top_p, max_tokens });
    try {
      stream = await client.chat.completions.create(payload);
    } catch (err) {
      const msg = err?.message || String(err);
      const lower = msg.toLowerCase();
      const isModel404 = lower.includes('does not exist') || lower.includes('model_not_found');
      if (isModel404) {
        log('warn','cerebras.stream.model_not_found', { id: req._reqId, model, msg });
        return res.status(404).json({ error: { message: msg, code: 'model_not_found' } });
      }
      log('error','cerebras.stream.upstream_error', { id: req._reqId, model, msg });
      return res.status(502).json({ error: { message: msg, code: 'upstream_error' } });
    }
  } catch (e) {
    const msg = e?.message || String(e);
    log('error','cerebras.stream.init_error', { id: req._reqId, error: msg });
    return res.status(500).json({ error: { message: msg, code: 'internal_error' } });
  }
  // SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  let closed = false;
  req.on('close', () => { closed = true; });

  let chunks = 0;
  const started = Date.now();
  (async () => {
    try {
      for await (const chunk of stream) {
        if (closed) break;
        const delta = chunk?.choices?.[0]?.delta?.content || '';
        res.write(`data: ${JSON.stringify({ delta })}\n\n`);
        if (delta) chunks++;
      }
      if (!closed) res.write('data: {"done": true}\n\n');
      log('info','cerebras.stream.end', { id: req._reqId, model, ms: Date.now()-started, chunks });
    } catch (e) {
      log('error','cerebras.stream.error', { id: req._reqId, error: e?.message || String(e), chunks });
      if (!closed) res.write(`data: ${JSON.stringify({ error: e?.message || String(e) })}\n\n`);
    } finally {
      if (!closed) res.end();
    }
  })();
});

// ---------------------------- Mistral ROUTES (with continue/tools/instructions) ------------------------
app.post('/mistral/chat', async (req, res) => {
  let { model, messages, temperature = 0.7, top_p = 1, max_tokens = 2048, apiKey, instructions = '', tools = null, continue: doContinue = false } = req.body || {};
  max_tokens = capTokens(max_tokens);
  res.setHeader('x-request-id', req._reqId || '');
  try {
    if (!model) return res.status(400).json({ error: { message: 'model required', code: 'model_required' } });
    const key = apiKey || process.env.MISTRAL_API_KEY;
    const client = buildMistralClient(key);
  let inputs = normalizeMessages(messages).map(m => ({ role: m.role, content: m.content }));
  inputs = transformForMistralVision(inputs);
    // Mistral currently does not accept image/document multimodal inputs in this deployment.
    // If any transformed part contains a documentUrl (from image_url), reject early with a helpful error.
    const hasDocument = inputs.some(msg => Array.isArray(msg.content) && msg.content.some(p => p && (p.documentUrl || p.type === 'document_url')));
    if (hasDocument) {
      return res.status(400).json({ error: { message: 'Mistral model selected does not support images/documents in this deployment. Upload images to a supported vision model or remove attachments.', code: 'mistral_vision_unsupported' } });
    }
    if (doContinue) inputs = inputs.concat([{ role: 'user', content: 'Continue.' }]);
    const t0 = Date.now();
    let response;
    try {
      response = await client.chat.complete({
        model,
        messages: inputs,
        temperature,
        top_p,
        max_tokens,
        instructions: instructions || undefined,
        tools: Array.isArray(tools) ? tools : undefined,
      });
    } catch (err) {
      const msg = err?.message || String(err);
      const lower = msg.toLowerCase();
      const isModel404 = lower.includes('not found') || lower.includes('model_not_found');
      if (isModel404) {
        log('warn','mistral.model_not_found', { id: req._reqId, model, msg });
        return res.status(404).json({ error: { message: msg, code: 'model_not_found' } });
      }
      log('error','mistral.upstream_error', { id: req._reqId, model, msg });
      return res.status(502).json({ error: { message: msg, code: 'upstream_error' } });
    }
    const latency_ms = Date.now() - t0;
    const usage = response?.usage || null;
    log('info','mistral.response', { id: req._reqId, model, latency_ms, usage });
    res.json({ ...response, latency_ms, request_id: req._reqId });
  } catch (e) {
    const msg = e?.message || String(e);
    log('error','mistral.error', { id: req._reqId, error: msg });
    res.status(500).json({ error: { message: msg, code: 'internal_error' } });
  }
});

app.post('/mistral/chat/stream', async (req, res) => {
  let { model, messages, temperature = 0.7, top_p = 1, max_tokens = 2048, apiKey, instructions = '', tools = null, continue: doContinue = false } = req.body || {};
  max_tokens = capTokens(max_tokens);
  res.setHeader('x-request-id', req._reqId || '');
  if (!model) return res.status(400).json({ error: { message: 'model required', code: 'model_required' } });
  let stream;
  try {
    const key = apiKey || process.env.MISTRAL_API_KEY;
    const client = buildMistralClient(key);
    let inputs = normalizeMessages(messages).map(m => ({ role: m.role, content: m.content }));
    inputs = transformForMistralVision(inputs);
      // Mistral does not support vision inputs here. If any part is a documentUrl (image), reject early.
      const hasDoc = inputs.some(msg => Array.isArray(msg.content) && msg.content.some(p => p && (p.documentUrl || p.type === 'document_url')));
      if (hasDoc) {
        return res.status(400).json({ error: { message: 'Mistral streaming endpoint does not accept images/documents in this deployment. Use a vision-capable model or remove attachments.', code: 'mistral_vision_unsupported' } });
      }
    if (doContinue) inputs = inputs.concat([{ role: 'user', content: 'Continue.' }]);
    try {
      stream = await client.chat.stream({
        model,
        messages: inputs,
        temperature,
        top_p,
        max_tokens,
        instructions: instructions || undefined,
        tools: Array.isArray(tools) ? tools : undefined,
      });
    } catch (err) {
      const msg = err?.message || String(err);
      const lower = msg.toLowerCase();
      const isModel404 = lower.includes('not found') || lower.includes('model_not_found');
      if (isModel404) {
        log('warn','mistral.stream.model_not_found', { id: req._reqId, model, msg });
        return res.status(404).json({ error: { message: msg, code: 'model_not_found' } });
      }
      log('error','mistral.stream.upstream_error', { id: req._reqId, model, msg });
      return res.status(502).json({ error: { message: msg, code: 'upstream_error' } });
    }
  } catch (e) {
    const msg = e?.message || String(e);
    log('error','mistral.stream.init_error', { id: req._reqId, error: msg });
    return res.status(500).json({ error: { message: msg, code: 'internal_error' } });
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  let closed = false;
  req.on('close', () => { closed = true; });

  let chunks = 0;
  const started = Date.now();
  (async () => {
    try {
      for await (const event of stream) {
        if (closed) break;
        try {
          const choice = event?.choices?.[0];
          let raw = choice?.delta?.content || choice?.message?.content || '';
          // If raw is an array of parts, flatten to text
          const deltaText = flattenPartsToText(raw);
          res.write(`data: ${JSON.stringify({ delta: deltaText })}\n\n`);
          if (deltaText) chunks++;
        } catch (inner) {
          res.write(`data: ${JSON.stringify({ error: inner?.message || String(inner) })}\n\n`);
        }
      }
      if (!closed) res.write('data: {"done": true}\n\n');
      log('info','mistral.stream.end', { id: req._reqId, model, ms: Date.now()-started, chunks });
    } catch (e) {
      log('error','mistral.stream.error', { id: req._reqId, error: e?.message || String(e), chunks });
      if (!closed) res.write(`data: ${JSON.stringify({ error: e?.message || String(e) })}\n\n`);
    } finally {
      if (!closed) res.end();
    }
  })();
});

// ---------------------------- Gemini ROUTES ------------------------
app.post('/gemini/chat', async (req, res) => {
  let { model = 'gemini-2.5-flash', messages, temperature = 0.7, top_p = 0.95, max_tokens = 8192, apiKey } = req.body || {};
  max_tokens = capTokens(max_tokens);
  res.setHeader('x-request-id', req._reqId || '');
  try {
    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: { message: 'messages array required', code: 'messages_required' } });
    }
    
    const key = apiKey || process.env.GEMINI_API_KEY;
    const client = buildGeminiClient(key);
    
    // Convert OpenAI-style messages to Gemini contents format
    const contents = convertToGeminiContents(messages);
    
    const t0 = Date.now();
    log('debug', 'gemini.request', { id: req._reqId, model, stream: false, temp: temperature, top_p, max_tokens });
    
    let response;
    try {
      // Gemini SDK requires 'models/' prefix if not already present
      const modelName = model.startsWith('models/') ? model : `models/${model}`;
      response = await client.models.generateContent({
        model: modelName,
        contents,
      });
    } catch (err) {
      const msg = err?.message || String(err);
      const lower = msg.toLowerCase();
      const isModel404 = lower.includes('not found') || lower.includes('model_not_found') || lower.includes('does not exist');
      if (isModel404) {
        log('warn', 'gemini.model_not_found', { id: req._reqId, model, msg });
        return res.status(404).json({ error: { message: msg, code: 'model_not_found' } });
      }
      log('error', 'gemini.upstream_error', { id: req._reqId, model, msg });
      return res.status(502).json({ error: { message: msg, code: 'upstream_error' } });
    }
    
    const latency_ms = Date.now() - t0;
    const text = response?.text || '';
    
    log('info', 'gemini.response', { id: req._reqId, model, latency_ms, textLength: text.length });
    
    // Format response in OpenAI-compatible structure
    res.json({
      id: req._reqId,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: text,
        },
        finish_reason: 'stop',
      }],
      latency_ms,
      request_id: req._reqId,
    });
  } catch (e) {
    const msg = e?.message || String(e);
    log('error', 'gemini.error', { id: req._reqId, error: msg });
    res.status(500).json({ error: { message: msg, code: 'internal_error' } });
  }
});

app.post('/gemini/chat/stream', async (req, res) => {
  let { model = 'gemini-2.5-flash', messages, temperature = 0.7, top_p = 0.95, max_tokens = 8192, apiKey } = req.body || {};
  max_tokens = capTokens(max_tokens);
  res.setHeader('x-request-id', req._reqId || '');
  
  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: { message: 'messages array required', code: 'messages_required' } });
  }
  
  let stream;
  try {
    const key = apiKey || process.env.GEMINI_API_KEY;
    const client = buildGeminiClient(key);
    
    // Convert OpenAI-style messages to Gemini contents format
    const contents = convertToGeminiContents(messages);
    
    log('debug', 'gemini.stream.begin', { id: req._reqId, model, temp: temperature, top_p, max_tokens });
    
    try {
      // Gemini SDK requires 'models/' prefix if not already present
      const modelName = model.startsWith('models/') ? model : `models/${model}`;
      stream = await client.models.generateContentStream({
        model: modelName,
        contents,
      });
    } catch (err) {
      const msg = err?.message || String(err);
      const lower = msg.toLowerCase();
      const isModel404 = lower.includes('not found') || lower.includes('model_not_found') || lower.includes('does not exist');
      if (isModel404) {
        log('warn', 'gemini.stream.model_not_found', { id: req._reqId, model, msg });
        return res.status(404).json({ error: { message: msg, code: 'model_not_found' } });
      }
      log('error', 'gemini.stream.upstream_error', { id: req._reqId, model, msg });
      return res.status(502).json({ error: { message: msg, code: 'upstream_error' } });
    }
  } catch (e) {
    const msg = e?.message || String(e);
    log('error', 'gemini.stream.init_error', { id: req._reqId, error: msg });
    return res.status(500).json({ error: { message: msg, code: 'internal_error' } });
  }
  
  // SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no'); // Disable nginx buffering
  res.flushHeaders?.();
  
  let closed = false;
  req.on('close', () => { closed = true; });
  
  let chunks = 0;
  const started = Date.now();
  
  (async () => {
    try {
      for await (const chunk of stream) {
        if (closed) break;
        const delta = chunk?.text || '';
        if (delta) {
          res.write(`data: ${JSON.stringify({ delta })}\n\n`);
          // Force flush to client immediately
          if (res.flush) res.flush();
          chunks++;
        }
      }
      
      if (!closed) {
        res.write('data: {"done": true}\n\n');
        if (res.flush) res.flush();
      }
      log('info', 'gemini.stream.end', { id: req._reqId, model, ms: Date.now() - started, chunks });
    } catch (e) {
      log('error', 'gemini.stream.error', { id: req._reqId, error: e?.message || String(e), chunks });
      if (!closed) {
        res.write(`data: ${JSON.stringify({ error: e?.message || String(e) })}\n\n`);
        if (res.flush) res.flush();
      }
    } finally {
      if (!closed) res.end();
    }
  })();
});

const PORT = process.env.PORT || 5058;

// Self-ping mechanism to keep Render service active
function startSelfPing() {
  const PING_INTERVAL = 5 * 60 * 1000; // 5 minutes
  const pingUrl = process.env.RENDER_EXTERNAL_URL || 'https://switchai.onrender.com';
  
  async function selfPing() {
    try {
      const response = await fetch(`${pingUrl}/health`);
      if (response.ok) {
        log('info', 'self.ping.success', { url: pingUrl });
      } else {
        log('warn', 'self.ping.failed', { url: pingUrl, status: response.status });
      }
    } catch (error) {
      log('warn', 'self.ping.error', { url: pingUrl, error: error.message });
    }
  }
  
  // Start pinging after 30 seconds, then every 5 minutes
  setTimeout(() => {
    selfPing(); // First ping
    setInterval(selfPing, PING_INTERVAL); // Regular pings
    log('info', 'self.ping.started', { interval: PING_INTERVAL, url: pingUrl });
  }, 30000);
}

app.listen(PORT, () => {
  log('info','server.start',{ port: PORT, logLevel: LOG_LEVEL });
  
  // Start self-ping only in production (Render) environment
  if (process.env.RENDER_EXTERNAL_URL || process.env.NODE_ENV === 'production') {
    startSelfPing();
  }
});
