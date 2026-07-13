// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from 'vitest'
import {
  splitReferences,
  tabularizeEntityLines,
  renumberOrderedLists,
  nestBulletsUnderOrderedItems,
  stripTrailingReferences,
  extractSchemaTables,
} from './parse-references'

describe('extractSchemaTables', () => {
  const diagram = [
    'Here is the schema:',
    '',
    '```mermaid',
    'erDiagram',
    '  customers ||--o{ sales_orders : places',
    '  sales_orders ||--|{ order_lines : contains',
    '  customers {',
    '    string customer_id',
    '    string segment',
    '  }',
    '  sales_orders {',
    '    string order_id',
    '  }',
    '  order_lines {',
    '    string line_id',
    '  }',
    '```',
  ].join('\n')

  test('pulls table names from entity blocks and relationships, sorted and de-duped', () => {
    expect(extractSchemaTables(diagram)).toEqual(['customers', 'order_lines', 'sales_orders'])
  })

  test('returns empty for content with no mermaid ER diagram', () => {
    expect(extractSchemaTables('just a normal answer with no diagram')).toEqual([])
    expect(extractSchemaTables('```mermaid\nflowchart TD\n A --> B\n```')).toEqual([])
  })
})

describe('nestBulletsUnderOrderedItems', () => {
  test('indents bullets that follow a numbered item so they nest under it', () => {
    const md = '1. Top drivers:\n- Schumacher: 91\n- Prost: 51'
    expect(nestBulletsUnderOrderedItems(md)).toBe('1. Top drivers:\n   - Schumacher: 91\n   - Prost: 51')
  })

  test('leaves top-level bullets with no preceding numbered item alone', () => {
    const md = '- a\n- b'
    expect(nestBulletsUnderOrderedItems(md)).toBe(md)
  })

  test('stops nesting at a heading', () => {
    const md = '1. x\n- under x\n## Section\n- top level'
    expect(nestBulletsUnderOrderedItems(md)).toBe('1. x\n   - under x\n## Section\n- top level')
  })

  test('re-indents an under-indented (1-2 space) bullet to nest under the item', () => {
    const md = '2. Categories:\n  - Garment: $1\n  - Swimwear: $2'
    expect(nestBulletsUnderOrderedItems(md)).toBe('2. Categories:\n   - Garment: $1\n   - Swimwear: $2')
  })

  test('leaves an already-nested (3+ space) bullet untouched', () => {
    const md = '2. Categories:\n   - Garment: $1\n      - deeper'
    expect(nestBulletsUnderOrderedItems(md)).toBe(md)
  })
})

describe('stripTrailingReferences', () => {
  test('strips a trailing `## Sources` section ([N] Title: URL format)', () => {
    const md =
      'The report is complete [1].\n\n## Sources\n[1] Q4 report: https://x.com/r\n[2] Internal notes: knowledge_search'
    expect(stripTrailingReferences(md)).toBe('The report is complete [1].')
  })

  test('strips a `### References` heading section', () => {
    const md = 'Body text [1].\n\n### References\n[1] fleet_overview.pdf, p.4'
    expect(stripTrailingReferences(md)).toBe('Body text [1].')
  })

  test('strips a bold `**Sources**` label section', () => {
    const md = 'Body.\n\n**Sources**\n[1] thing - https://a.example.com'
    expect(stripTrailingReferences(md)).toBe('Body.')
  })

  test('strips even when the listed lines are not [N]-parseable', () => {
    const md = 'Body.\n\n## Sources\n1 Internal KB: schema\n7 Internal KB: knowledge_search'
    expect(stripTrailingReferences(md)).toBe('Body.')
  })

  test('leaves a body with no references section unchanged', () => {
    const md = 'Just a report.\n\n## Conclusion\nAll good.'
    expect(stripTrailingReferences(md)).toBe(md)
  })
})

describe('renumberOrderedLists', () => {
  test('renumbers top-level items that all restart at 1', () => {
    const md = '1. Counts:\n- a: 1\n- b: 2\n1. Top drivers:\n- x\n1. Summary:'
    expect(renumberOrderedLists(md)).toBe('1. Counts:\n- a: 1\n- b: 2\n2. Top drivers:\n- x\n3. Summary:')
  })

  test('resets numbering at each heading so sections keep 1-based order', () => {
    const md = '## A\n1. one\n1. two\n## B\n1. three'
    expect(renumberOrderedLists(md)).toBe('## A\n1. one\n2. two\n## B\n1. three')
  })

  test('leaves already-sequential lists and nested items untouched', () => {
    const md = '1. one\n2. two\n   1. nested'
    expect(renumberOrderedLists(md)).toBe(md)
  })
})

describe('splitReferences', () => {
  test('returns the body unchanged when there is no references block', () => {
    const content = 'The fleet has 11,463 GPUs across 12 clusters.'
    const { body, sources } = splitReferences(content)
    expect(body).toBe(content)
    expect(sources).toEqual([])
  })

  test('strips a **References:** block and parses web sources', () => {
    const content =
      'NVIDIA shipped record volume [1].\n\n**References:**\n- [1] NVIDIA Q4 results - https://www.nvidia.com/news\n- [2] Industry brief - https://docs.example.ai/report'
    const { body, sources } = splitReferences(content)
    expect(body).toBe('NVIDIA shipped record volume [1].')
    expect(sources).toHaveLength(2)
    expect(sources[0]).toMatchObject({
      title: 'NVIDIA Q4 results',
      url: 'https://www.nvidia.com/news',
      kind: 'web',
      label: 'nvidia.com',
    })
    expect(sources[1]).toMatchObject({ url: 'https://docs.example.ai/report', kind: 'web', label: 'docs.example.ai' })
  })

  test('parses file-with-page and bare tool references as documents', () => {
    const content =
      'See attached docs [1] and the search [2].\n\n## Sources\n- [1] fleet_overview.pdf, p.4\n- [2] knowledge_search'
    const { body, sources } = splitReferences(content)
    expect(body).toBe('See attached docs [1] and the search [2].')
    expect(sources[0]).toMatchObject({ kind: 'doc', title: 'fleet_overview.pdf, p.4' })
    expect(sources[1]).toMatchObject({ kind: 'doc', title: 'knowledge_search', label: 'knowledge_search' })
  })

  test('orders sources by their [N] index regardless of line order', () => {
    const content =
      'Body.\n\n**References**\n- [2] Second - https://b.example.com\n- [1] First - https://a.example.com'
    const { sources } = splitReferences(content)
    expect(sources.map((s) => s.title)).toEqual(['First', 'Second'])
  })

  test('leaves content intact when the block has no parseable lines', () => {
    const content = 'Body.\n\n**References:**\nnothing structured here'
    const { body, sources } = splitReferences(content)
    expect(body).toBe(content)
    expect(sources).toEqual([])
  })
})

describe('tabularizeEntityLines', () => {
  test('reflows 2+ consecutive label: value lines into a GFM table', () => {
    const body = 'Top teams by job count:\n\nteam_051: ~11,285 jobs\nteam_009: ~8,210 jobs\nteam_044: ~6,140 jobs'
    const out = tabularizeEntityLines(body)
    expect(out).toContain('| Item | Value |')
    expect(out).toContain('| --- | --- |')
    expect(out).toContain('| team_051 | ~11,285 jobs |')
    expect(out).toContain('| team_044 | ~6,140 jobs |')
  })

  test('leaves prose untouched', () => {
    const body = 'The result is clear: the fleet is healthy. Utilization rose this quarter.'
    expect(tabularizeEntityLines(body)).toBe(body)
  })

  test('does not tabularize a single label: value line', () => {
    const body = 'Summary: the fleet is healthy'
    expect(tabularizeEntityLines(body)).toBe(body)
  })

  test('does not touch markdown list items', () => {
    const body = '- team_051: ~11,285 jobs\n- team_009: ~8,210 jobs'
    expect(tabularizeEntityLines(body)).toBe(body)
  })

  test('copies fenced ```chart blocks through verbatim (no table from JSON)', () => {
    const body =
      'Distribution by model:\n\n```chart\n{\n  "type": "bar",\n  "series": [{ "key": "count" }],\n  "data": [{ "model": "H100", "count": 5120 }]\n}\n```\n\nH100 leads.'
    const out = tabularizeEntityLines(body)
    expect(out).toBe(body)
    expect(out).not.toContain('| Item | Value |')
    expect(out).toContain('"count": 5120')
  })

  test('renumber + nest-bullets leave fenced content untouched', () => {
    const body = '1. one\n```chart\n1. not-a-list-item\n- not-a-bullet\n```\n1. two'
    expect(renumberOrderedLists(body)).toBe('1. one\n```chart\n1. not-a-list-item\n- not-a-bullet\n```\n2. two')
    expect(nestBulletsUnderOrderedItems(body)).toBe(body)
  })
})
