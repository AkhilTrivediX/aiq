// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@/test-utils'
import userEvent from '@testing-library/user-event'
import { describe, test, expect } from 'vitest'
import { ChatThinking } from './ChatThinking'
import type { ThinkingStep } from '../types'

const createStep = (overrides: Partial<ThinkingStep> = {}): ThinkingStep => ({
  id: 'step-1',
  userMessageId: 'msg-1',
  category: 'tasks',
  functionName: 'web_search_tool',
  displayName: 'Searching the web',
  content: 'Step content here',
  isComplete: false,
  timestamp: new Date('2024-01-15T14:30:00'),
  ...overrides,
})

describe('ChatThinking', () => {
  describe('empty state', () => {
    test('renders nothing when no steps, sources, or files are provided', () => {
      render(<ChatThinking steps={[]} />)
      expect(screen.queryByLabelText('Working')).not.toBeInTheDocument()
      expect(screen.queryByText('Done')).not.toBeInTheDocument()
      expect(screen.queryByText(/\bstep\b/)).not.toBeInTheDocument()
    })
  })

  describe('status header', () => {
    test('shows the working spinner and a thinking word while thinking', () => {
      render(<ChatThinking steps={[createStep()]} isThinking />)

      expect(screen.getByLabelText('Working')).toBeInTheDocument()
      expect(screen.getByText('Thinking')).toBeInTheDocument()
    })

    test('shows Done when finished', () => {
      render(<ChatThinking steps={[createStep()]} isThinking={false} />)

      expect(screen.getByText('Done')).toBeInTheDocument()
      expect(screen.queryByLabelText('Working')).not.toBeInTheDocument()
    })

    test('shows Interrupted when interrupted', () => {
      render(<ChatThinking steps={[createStep()]} isThinking={false} isInterrupted />)

      expect(screen.getByText('Interrupted')).toBeInTheDocument()
      expect(screen.queryByText('Done')).not.toBeInTheDocument()
    })

    test('shows the waiting label when awaiting user input', () => {
      render(<ChatThinking steps={[createStep()]} isThinking={false} isWaiting />)

      expect(screen.getByText('Needs your input')).toBeInTheDocument()
      expect(screen.queryByText('Done')).not.toBeInTheDocument()
    })

    test('thinking takes priority over interrupted', () => {
      render(<ChatThinking steps={[createStep()]} isThinking isInterrupted />)

      expect(screen.getByLabelText('Working')).toBeInTheDocument()
      expect(screen.queryByText('Interrupted')).not.toBeInTheDocument()
    })
  })

  describe('phase trace', () => {
    test('shows the step count when collapsed', () => {
      render(<ChatThinking steps={[createStep()]} isThinking={false} />)

      expect(screen.getByText('1 step')).toBeInTheDocument()
    })

    test('renders the human tool label in the trace while thinking', () => {
      render(<ChatThinking steps={[createStep({ functionName: 'web_search_tool' })]} isThinking />)

      expect(screen.getByText('Searching the web')).toBeInTheDocument()
    })

    test('expands the collapsed trace on click', async () => {
      const user = userEvent.setup()
      render(
        <ChatThinking steps={[createStep({ functionName: 'paper_search_tool' })]} isThinking={false} />
      )

      await user.click(screen.getByText('1 step'))

      expect(screen.getByText('Searching papers')).toBeInTheDocument()
    })

    test('folds reasoning and explanation notes into a phase instead of new rows', () => {
      const steps = [
        createStep({ id: 'a', functionName: 'web_search_tool', isTopLevel: true }),
        createStep({ id: 'b', functionName: '__reasoning__', content: 'weighing the options' }),
        createStep({ id: 'c', functionName: '__explanation__', content: 'why this matters' }),
      ]
      render(<ChatThinking steps={steps} isThinking={false} />)

      // Only the tool phase is counted; the folded notes are not their own steps.
      expect(screen.getByText('1 step')).toBeInTheDocument()
    })
  })

  describe('used sources and files', () => {
    test('surfaces a Using chip for a source the answer actually used', () => {
      render(
        <ChatThinking
          steps={[createStep({ functionName: 'web_search_tool' })]}
          isThinking={false}
          enabledDataSources={['web_search']}
        />
      )

      expect(screen.getByText('Using')).toBeInTheDocument()
      expect(screen.getByText('Web Search')).toBeInTheDocument()
    })

    test('does not surface an enabled source that no tool call used', () => {
      render(
        <ChatThinking
          steps={[createStep({ functionName: 'web_search_tool' })]}
          isThinking={false}
          enabledDataSources={['web_search', 'confluence']}
        />
      )

      expect(screen.getByText('Web Search')).toBeInTheDocument()
      expect(screen.queryByText('Confluence')).not.toBeInTheDocument()
    })

    test('shows message files as source chips', () => {
      render(
        <ChatThinking
          steps={[createStep()]}
          isThinking={false}
          messageFiles={[{ id: 'f1', fileName: 'document.pdf' }]}
        />
      )

      expect(screen.getByText('document.pdf')).toBeInTheDocument()
    })
  })

  describe('embedded variant', () => {
    test('renders only the trace spine, without the chat header', () => {
      render(<ChatThinking steps={[createStep({ functionName: 'web_search_tool' })]} embedded />)

      expect(screen.getByText('Searching the web')).toBeInTheDocument()
      expect(screen.queryByText('Thinking')).not.toBeInTheDocument()
      expect(screen.queryByText('Done')).not.toBeInTheDocument()
    })
  })
})
