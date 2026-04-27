import express from 'express'

export function registerRoutes(app: express.Express) {
  app.get('/health', (_req, res) => {
    res.json({ ok: true })
  })
}
