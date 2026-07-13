// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, test } from 'vitest'
import { Citation } from './Citation'
import type { SourceRef } from '@/shared/components/Sources/types'

const sources: SourceRef[] = [
  {
    id: 's1',
    title: 'NVIDIA shipped record volume.',
    kind: 'web',
    label: 'nvidia.com',
    url: 'https://www.nvidia.com/news',
    snippet: 'Record shipments across the fleet.',
  },
]

describe('Citation', () => {
  test('renders the citation number with an accessible label', () => {
    render(<Citation n={1} sources={sources} />)
    expect(screen.getByLabelText('Source 1: nvidia.com')).toHaveTextContent('1')
  })

  test('renders the marker as a superscript reference mark', () => {
    render(<Citation n={1} sources={sources} />)
    const link = screen.getByLabelText('Source 1: nvidia.com')
    expect(link.querySelector('sup')).toHaveTextContent('1')
  })

  test('links to the source url', () => {
    render(<Citation n={1} sources={sources} />)
    expect(screen.getByLabelText('Source 1: nvidia.com')).toHaveAttribute(
      'href',
      'https://www.nvidia.com/news'
    )
  })

  test('a non-numeric value renders bracketed text and no chip', () => {
    render(<Citation n={Number.NaN} sources={sources} />)
    expect(screen.queryByRole('link')).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/^Source/)).not.toBeInTheDocument()
  })

  test('an out-of-range index degrades to a bare chip', () => {
    render(<Citation n={9} sources={sources} />)
    expect(screen.getByLabelText('Source 9')).toBeInTheDocument()
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  })

  test('focus reveals a popover with the source label and title', () => {
    render(<Citation n={1} sources={sources} />)
    fireEvent.focus(screen.getByLabelText('Source 1: nvidia.com'))
    const tip = screen.getByRole('tooltip')
    expect(tip).toHaveTextContent('nvidia.com')
    expect(tip).toHaveTextContent('NVIDIA shipped record volume.')
  })
})
