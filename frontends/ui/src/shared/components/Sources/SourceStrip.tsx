// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client'

import { type ReactNode, useState } from 'react'
import { Card } from '@/shared/components/Surface/Card'
import { cn } from '@/shared/lib/cn'
import { SourceKindIcon } from './SourceKindIcon'
import type { SourceRef } from './types'

interface SourceCardProps {
  source: SourceRef
  index: number
}

/**
 * One source rendered as a typed card: title, an origin icon + label, and the
 * citation index. Becomes a link when the source has a real URL.
 */
export function SourceCard({ source, index }: SourceCardProps): ReactNode {
  const body = (
    <Card
      tone={source.kind}
      className="hover:bg-surface-sunken flex h-full flex-col gap-2 p-3 transition-colors"
    >
      <p className="text-primary line-clamp-2 text-sm">{source.title}</p>
      <div className="text-subtle mt-auto flex items-center justify-between text-xs">
        <span className="text-secondary flex min-w-0 items-center gap-1.5">
          <SourceKindIcon kind={source.kind} />
          <span className="truncate">{source.label}</span>
        </span>
        <span aria-hidden="true">[{index + 1}]</span>
      </div>
    </Card>
  )
  if (source.url) {
    return (
      <a href={source.url} target="_blank" rel="noopener noreferrer" className="block no-underline">
        {body}
      </a>
    )
  }
  return body
}

interface SourceStripProps {
  sources: SourceRef[]
  /** How many cards to show before the "+N more" expander. */
  previewCount?: number
  className?: string
}

/**
 * Answer-attached strip of typed source cards. Shows a preview, then expands in
 * place to reveal the rest, keeping sources present but subordinate to the answer.
 */
export function SourceStrip({ sources, previewCount = 4, className }: SourceStripProps): ReactNode {
  const [expanded, setExpanded] = useState(false)
  if (sources.length === 0) return null

  const hasMore = sources.length > previewCount
  const visible = expanded || !hasMore ? sources : sources.slice(0, previewCount - 1)

  return (
    <section className={cn('space-y-2', className)} aria-label="Sources">
      <h3 className="text-secondary flex items-center gap-1.5 text-sm font-medium">
        <SourceKindIcon kind="doc" /> Sources
      </h3>
      <div className="flex flex-col gap-2">
        {visible.map((s, i) => (
          <SourceCard key={s.id} source={s} index={i} />
        ))}
        {hasMore && !expanded && (
          <button
            type="button"
            onClick={() => setExpanded(true)}
            aria-expanded={expanded}
            className="bg-surface-raised border-base text-secondary hover:bg-surface-sunken hover:text-primary flex flex-col items-start justify-center gap-1 rounded-[var(--radius-card)] border p-3 text-sm transition-colors"
          >
            <span className="font-medium">+{sources.length - (previewCount - 1)} more</span>
            <span className="text-subtle text-xs">View all sources</span>
          </button>
        )}
      </div>
    </section>
  )
}
