#!/usr/bin/env node
/*
  SwitchAI Cerebras Proxy Server
  --------------------------------
  Endpoints:
    GET  /health                  -> { status: 'ok', ts }
    POST /cerebras/chat           -> Non-stream JSON completion
      Body: { model, messages, temperature?, top_p?, max_tokens?, apiKey?, useCase? }
    POST /cerebras/chat/stream    -> Server-Sent Events streaming
      Body: { model, messages, temperature?, top_p?, max_tokens?, apiKey?, useCase? }

  apiKey precedence: request body apiKey > process.env.CEREBRAS_API_KEY
  
  Dynamic Token Limits:
    - Default (chat): max 32,768 tokens
    - Extended (audio-analysis, pdf-analysis): max 50,000 tokens
    - Set useCase to 'audio-analysis' or 'pdf-analysis' for extended limits

  Example non-stream request:
    curl -X POST http://localhost:5058/cerebras/chat -H 'Content-Type: application/json' \
      -d '{"model":"qwen-3-coder-480b","messages":[{"role":"user","content":"Hello"}]}'

  Example extended tokens request:
    curl -X POST http://localhost:5058/cerebras/chat -H 'Content-Type: application/json' \
      -d '{"model":"qwen-3-235b-a22b-instruct-2507","messages":[...],"max_tokens":45000,"useCase":"pdf-analysis"}'
*/
import Cerebras from '@cerebras/cerebras_cloud_sdk';
import cors from 'cors';
import crypto from 'crypto';
import 'dotenv/config';
import express from 'express';

const app = express();
app.use(cors({ origin: '*', maxAge: 600 }));
app.use(express.json({ limit: '2mb' }));

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
  }
  res.on('finish', done); res.on('close', done); res.on('error', done);
  req._reqId = id; // attach for downstream use
  next();
});

function buildClient(apiKey) {
  if (!apiKey) throw new Error('Missing Cerebras API key');
  return new Cerebras({ apiKey });
}

function normalizeMessages(raw) {
  if (!Array.isArray(raw)) return [];
  return raw.map(m => ({ role: m.role || 'user', content: m.content || '' }));
}

app.get('/health', (req, res) => {
  res.json({ status: 'ok', ts: Date.now() });
});

// Dynamic max tokens based on use case
const DEFAULT_MAX_TOKENS = 32768;  // For general chat
const EXTENDED_MAX_TOKENS = 50000; // For audio/PDF analysis
const ABSOLUTE_HARD_MAX = 128000;  // Cerebras absolute limit

function capTokens(v, allowExtended = false){
  if (typeof v !== 'number' || !isFinite(v) || v <= 0) return 4096;
  const hardMax = allowExtended ? EXTENDED_MAX_TOKENS : DEFAULT_MAX_TOKENS;
  return Math.min(ABSOLUTE_HARD_MAX, Math.max(1, Math.floor(Math.min(v, hardMax))));
}

app.post('/cerebras/chat', async (req, res) => {
  let { model, messages, temperature = 0.7, top_p = 0.8, max_tokens = 8192, apiKey, useCase } = req.body || {};
  
  // Allow extended tokens for audio/PDF analysis use cases
  const allowExtended = useCase === 'audio-analysis' || useCase === 'pdf-analysis';
  max_tokens = capTokens(max_tokens, allowExtended);
  
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
  let { model, messages, temperature = 0.7, top_p = 0.8, max_tokens = 8192, apiKey, useCase } = req.body || {};
  
  // Allow extended tokens for audio/PDF analysis use cases  
  const allowExtended = useCase === 'audio-analysis' || useCase === 'pdf-analysis';
  max_tokens = capTokens(max_tokens, allowExtended);
  
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

const PORT = process.env.PORT || 5058;
app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  log('info','server.start',{ port: PORT, logLevel: LOG_LEVEL });
});
