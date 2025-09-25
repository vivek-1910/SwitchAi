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

  apiKey precedence: request body apiKey > process.env.CEREBRAS_API_KEY

  Example non-stream request:
    curl -X POST http://localhost:5058/cerebras/chat -H 'Content-Type: application/json' \
      -d '{"model":"qwen-3-coder-480b","messages":[{"role":"user","content":"Hello"}]}'

  Example stream request:
    curl -N -X POST http://localhost:5058/cerebras/chat/stream -H 'Content-Type: application/json' \
      -d '{"model":"qwen-3-coder-480b","messages":[{"role":"user","content":"Hello"}]}'
*/
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import Cerebras from '@cerebras/cerebras_cloud_sdk';
import crypto from 'crypto';

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

app.post('/cerebras/chat', async (req, res) => {
  const { model, messages, temperature = 0.7, top_p = 0.8, max_tokens = 8192, apiKey } = req.body || {};
  try {
    if (!model) return res.status(400).json({ error: 'model required' });
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
    const data = await client.chat.completions.create(payload);
    const latency_ms = Date.now() - t0;
    log('info','cerebras.response', { id: req._reqId, model, latency_ms, usage: data?.usage });
    res.json({ ...data, latency_ms });
  } catch (e) {
    // eslint-disable-next-line no-console
    log('error','cerebras.error', { id: req._reqId, error: e?.message || String(e) });
    res.status(500).json({ error: e?.message || String(e) });
  }
});

app.post('/cerebras/chat/stream', async (req, res) => {
  const { model, messages, temperature = 0.7, top_p = 0.8, max_tokens = 8192, apiKey } = req.body || {};
  if (!model) return res.status(400).json({ error: 'model required' });
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
    stream = await client.chat.completions.create(payload);
  } catch (e) {
    // eslint-disable-next-line no-console
    log('error','cerebras.stream.init_error', { id: req._reqId, error: e?.message || String(e) });
    return res.status(500).json({ error: e?.message || String(e) });
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
