// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Proxy-level SSE reconnect tests.
 *
 * EventSource sends the resume cursor in the Last-Event-ID request header on
 * automatic reconnect. The jobs proxy must forward that header to the backend
 * stream route; dropping it makes every reconnect replay from zero, which
 * duplicates UUID-backed progress entries in the UI.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'

vi.mock('next/headers', () => ({
  cookies: vi.fn(async () => ({ get: () => undefined })),
}))

vi.mock('@/adapters/auth/config', () => ({
  isAuthRequired: () => false,
}))

vi.mock('next/server', () => {
  class MockNextResponse extends Response {
    static json(body: unknown, init?: ResponseInit): Response {
      return new Response(JSON.stringify(body), {
        ...init,
        headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
      })
    }
  }
  return { NextResponse: MockNextResponse }
})

import { GET } from './route'

const sseBody = (): ReadableStream<Uint8Array> =>
  new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('id: 7\nevent: job.event\ndata: {}\n\n'))
      controller.close()
    },
  })

describe('jobs proxy SSE reconnect', () => {
  const fetchMock = vi.fn()

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock)
    fetchMock.mockResolvedValue(
      new Response(sseBody(), { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
    )
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  const call = async (path: string[], headers: Record<string, string> = {}): Promise<Response> => {
    const req = new Request(`http://localhost/api/jobs/async/${path.join('/')}`, { headers })
    return GET(req, { params: Promise.resolve({ path }) })
  }

  test('forwards Last-Event-ID to the backend on stream reconnect', async () => {
    await call(['job', 'job-1', 'stream'], { 'Last-Event-ID': '42' })

    expect(fetchMock).toHaveBeenCalledTimes(1)
    const init = fetchMock.mock.calls[0][1] as RequestInit
    expect((init.headers as Record<string, string>)['Last-Event-ID']).toBe('42')
  })

  test('omits Last-Event-ID on a fresh stream connection', async () => {
    await call(['job', 'job-1', 'stream'])

    const init = fetchMock.mock.calls[0][1] as RequestInit
    expect(init.headers as Record<string, string>).not.toHaveProperty('Last-Event-ID')
  })

  test('does not forward Last-Event-ID on non-stream requests', async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ status: 'RUNNING' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    )
    await call(['job', 'job-1'], { 'Last-Event-ID': '42' })

    const init = fetchMock.mock.calls[0][1] as RequestInit
    expect(init.headers as Record<string, string>).not.toHaveProperty('Last-Event-ID')
  })
})
