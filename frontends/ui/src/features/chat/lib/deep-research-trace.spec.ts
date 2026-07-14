// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from 'vitest'
import { deepResearchToThinkingSteps } from './deep-research-trace'
import { EXPLANATION_FUNCTION_NAME } from './intermediate-step-parser'
import type { DeepResearchAgent, DeepResearchToolCall } from '../types'

const agent = (o: Partial<DeepResearchAgent> = {}): DeepResearchAgent => ({
  id: 'a1',
  name: 'researcher',
  status: 'complete',
  startedAt: new Date('2026-06-25T00:00:00Z'),
  ...o,
})

const tool = (o: Partial<DeepResearchToolCall> = {}): DeepResearchToolCall => ({
  id: 't1',
  name: 'web_search_tool',
  status: 'complete',
  timestamp: new Date('2026-06-25T00:00:05Z'),
  ...o,
})

describe('deepResearchToThinkingSteps', () => {
  test('agents become top-level phase heads in start order, with input as the summary', () => {
    const steps = deepResearchToThinkingSteps(
      [
        agent({
          id: 'b',
          name: 'writer',
          startedAt: new Date('2026-06-25T00:01:00Z'),
          input: 'write the report',
        }),
        agent({
          id: 'a',
          name: 'planner',
          startedAt: new Date('2026-06-25T00:00:00Z'),
          input: 'plan the work',
        }),
      ],
      []
    )
    const heads = steps.filter((s) => s.isTopLevel)
    expect(heads.map((s) => s.functionName)).toEqual(['planner', 'writer'])
    expect(heads[0].argSummary).toBe('plan the work')
    expect(heads[0].status).toBe('success')
    expect(heads[0].isComplete).toBe(true)
  })

  test('tool calls nest under their agent with an arg summary', () => {
    const steps = deepResearchToThinkingSteps(
      [agent({ id: 'a1' })],
      [
        tool({
          id: 't1',
          name: 'web_search_tool',
          agentId: 'a1',
          input: { query: 'top customers by revenue' },
          output: 'results...',
        }),
      ]
    )
    const child = steps.find((s) => s.id === 't1')
    expect(child?.isTopLevel).toBe(false)
    expect(child?.argSummary).toBe('top customers by revenue')
  })

  test('an explain tool folds its output in as a __explanation__ step', () => {
    const steps = deepResearchToThinkingSteps(
      [agent({ id: 'a1' })],
      [tool({ id: 't1', name: 'explain_findings', agentId: 'a1', output: 'Key drivers: recent usage.' })]
    )
    const explanation = steps.find((s) => s.functionName === EXPLANATION_FUNCTION_NAME)
    expect(explanation?.content).toBe('Key drivers: recent usage.')
  })

  test('does not surface model reasoning in the deep trace', () => {
    const steps = deepResearchToThinkingSteps(
      [agent({ id: 'a1', name: 'planner' })],
      [tool({ id: 't1', name: 'web_search_tool', agentId: 'a1', output: 'ok' })]
    )
    expect(steps.some((s) => s.functionName === '__reasoning__')).toBe(false)
  })

  test('tool calls with no owning agent become their own top-level phases', () => {
    const steps = deepResearchToThinkingSteps([], [tool({ id: 't1', agentId: undefined })])
    expect(steps).toHaveLength(1)
    expect(steps[0].isTopLevel).toBe(true)
    expect(steps[0].functionName).toBe('web_search_tool')
  })

  test('maps a running run to a running, incomplete step', () => {
    const steps = deepResearchToThinkingSteps([agent({ status: 'running' })], [])
    expect(steps[0].status).toBe('running')
    expect(steps[0].isComplete).toBe(false)
  })

  test('maps an errored agent to an error step', () => {
    const steps = deepResearchToThinkingSteps([agent({ status: 'error' })], [])
    expect(steps[0].status).toBe('error')
  })

  test('collapses a run of identical retried tool calls into one row', () => {
    const steps = deepResearchToThinkingSteps(
      [agent({ id: 'a1' })],
      [
        tool({ id: 'r1', name: 'web_search_tool', agentId: 'a1', timestamp: new Date('2026-06-25T00:00:01Z') }),
        tool({ id: 'r2', name: 'web_search_tool', agentId: 'a1', timestamp: new Date('2026-06-25T00:00:02Z') }),
        tool({ id: 'r3', name: 'web_search_tool', agentId: 'a1', timestamp: new Date('2026-06-25T00:00:03Z') }),
      ]
    )
    const children = steps.filter((s) => !s.isTopLevel)
    expect(children).toHaveLength(1)
  })

  test('never merges agent phase heads, even with the same name', () => {
    const steps = deepResearchToThinkingSteps(
      [
        agent({ id: 'a1', name: 'researcher', startedAt: new Date('2026-06-25T00:00:01Z') }),
        agent({ id: 'a2', name: 'researcher', startedAt: new Date('2026-06-25T00:00:02Z') }),
      ],
      []
    )
    expect(steps.filter((s) => s.isTopLevel)).toHaveLength(2)
  })

  test('does NOT collapse tool calls with different arg summaries', () => {
    const steps = deepResearchToThinkingSteps(
      [agent({ id: 'a1' })],
      [
        tool({ id: 's1', name: 'web_search_tool', agentId: 'a1', input: { query: 'first' }, timestamp: new Date('2026-06-25T00:00:01Z') }),
        tool({ id: 's2', name: 'web_search_tool', agentId: 'a1', input: { query: 'second' }, timestamp: new Date('2026-06-25T00:00:02Z') }),
      ]
    )
    expect(steps.filter((s) => !s.isTopLevel)).toHaveLength(2)
  })

  test("orders an agent's tool calls by timestamp", () => {
    const steps = deepResearchToThinkingSteps(
      [agent({ id: 'a1' })],
      [
        tool({ id: 'late', name: 'web_search_tool', agentId: 'a1', input: { query: 'second' }, timestamp: new Date('2026-06-25T00:00:09Z') }),
        tool({ id: 'early', name: 'web_search_tool', agentId: 'a1', input: { query: 'first' }, timestamp: new Date('2026-06-25T00:00:03Z') }),
      ]
    )
    const childIds = steps.filter((s) => !s.isTopLevel).map((s) => s.id)
    expect(childIds).toEqual(['early', 'late'])
  })

  test('preserves a tool call whose referenced agent is missing', () => {
    const steps = deepResearchToThinkingSteps(
      [agent({ id: 'a1' })],
      [tool({ id: 'orphan', name: 'web_search_tool', agentId: 'ghost' })]
    )
    const orphan = steps.find((s) => s.id === 'orphan')
    expect(orphan).toBeDefined()
    expect(orphan?.isTopLevel).toBe(true)
  })
})
