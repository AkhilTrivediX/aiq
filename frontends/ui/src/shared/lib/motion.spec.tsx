// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, test, vi } from 'vitest'
import { render, screen } from '@/test-utils'
import { AppMotionConfig } from './motion'

const useReducedMotion = vi.hoisted(() => vi.fn(() => false))
vi.mock('@/hooks/use-reduced-motion', () => ({ useReducedMotion }))

describe('AppMotionConfig', () => {
  afterEach(() => vi.clearAllMocks())

  test('renders its children when motion is allowed', () => {
    useReducedMotion.mockReturnValue(false)
    render(
      <AppMotionConfig>
        <span>content</span>
      </AppMotionConfig>,
    )
    expect(screen.getByText('content')).toBeInTheDocument()
  })

  test('renders its children when reduced motion is preferred', () => {
    useReducedMotion.mockReturnValue(true)
    render(
      <AppMotionConfig>
        <span>reduced</span>
      </AppMotionConfig>,
    )
    expect(screen.getByText('reduced')).toBeInTheDocument()
  })
})
