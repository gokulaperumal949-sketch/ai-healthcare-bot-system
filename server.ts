import express, { Request, Response } from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const DEFAULT_PORT = Number(process.env.PORT) || 3000;

app.use(express.json());

// Initialize Google Gen AI client if key exists
const apiKey = process.env.GEMINI_API_KEY || '';
let aiClient: GoogleGenAI | null = null;
if (apiKey) {
  try {
    aiClient = new GoogleGenAI({ apiKey });
  } catch (err) {
    console.warn('Could not initialize GoogleGenAI:', err);
  }
}

// API Health Check
app.get('/api/health', (_req: Request, res: Response) => {
  res.json({
    status: 'ok',
    hasGeminiKey: Boolean(apiKey),
    timestamp: new Date().toISOString()
  });
});

// AI Healthcare Chat Route (Gemini 3.8 Flash)
app.post('/api/chat', async (req: Request, res: Response) => {
  try {
    const { message, history = [], userProfile } = req.body;

    if (!message || typeof message !== 'string') {
      res.status(400).json({ error: 'Message is required' });
      return;
    }

    if (!aiClient) {
      res.status(503).json({
        error: 'GEMINI_API_KEY not configured on server',
        fallbackNeeded: true
      });
      return;
    }

    const systemInstruction = `You are "ArogyaBot", an intelligent clinical AI Healthcare assistant designed for rural and urban community care.
Your core objectives:
1. Provide clear, empathetic, evidence-based medical information, symptom analysis, and immediate practical precautions.
2. Structure your reply with:
   - **Clinical Summary / Possible Causes** (non-diagnostic; always phrase as possibilities to discuss with a physician)
   - **Home Care & Precautionary Steps** (actionable, safe remedies, dietary and rest advice)
   - **Warning Red Flags** (symptoms requiring immediate ER / doctor consultation)
   - **Urgency Level**: [Routine | Needs Medical Attention | Emergency]
3. If life-threatening conditions (chest pain radiating to arm/jaw, sudden slurred speech/facial droop, severe breathlessness, profuse bleeding, anaphylaxis) are detected, trigger an **EMERGENCY ALERT** in your reply and advise dialing emergency services immediately (108/112/911).
4. Patient profile context (if provided): ${userProfile ? JSON.stringify(userProfile) : 'None'}.
5. Keep explanations direct, compassionate, and easy to understand. Never endorse unproven or dangerous treatments.`;

    const chatContents: Array<{ role: 'user' | 'model'; parts: Array<{ text: string }> }> = [];

    // Add recent history if available (limit to last 6 messages for context)
    const recentHistory = Array.isArray(history) ? history.slice(-6) : [];
    for (const msg of recentHistory) {
      if (msg.role === 'user' || msg.role === 'assistant') {
        chatContents.push({
          role: msg.role === 'user' ? 'user' : 'model',
          parts: [{ text: msg.content || '' }]
        });
      }
    }

    // Append the current message
    chatContents.push({
      role: 'user',
      parts: [{ text: message }]
    });

    const response = await aiClient.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: chatContents,
      config: {
        systemInstruction,
        temperature: 0.3,
        maxOutputTokens: 1000,
      }
    });

    const reply = response.text || 'I could not generate a response at this time. Please consult a doctor.';
    res.json({ reply, model: 'gemini-3.8-flash' });
  } catch (error: any) {
    console.error('Gemini API Error:', error);
    res.status(500).json({
      error: error?.message || 'Failed to process healthcare query',
      fallbackNeeded: true
    });
  }
});

async function startServer() {
  const isProd = process.env.NODE_ENV === 'production';

  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.resolve(distPath, 'index.html'));
    });
  }

  const startOnPort = (port: number) => {
    app.listen(port, '0.0.0.0', () => {
      console.log(`Healthcare Bot server running at http://0.0.0.0:${port}`);
    }).on('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'EADDRINUSE' && !process.env.PORT) {
        console.warn(`Port ${port} is busy, retrying on ${port + 1}...`);
        startOnPort(port + 1);
        return;
      }

      throw err;
    });
  };

  startOnPort(DEFAULT_PORT);
}

startServer();
