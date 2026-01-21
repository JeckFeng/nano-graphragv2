import { http } from './index'
import type { ConversationSummary, Message } from '@/types'

export const conversationApi = {
  create: (userId: string, title?: string) =>
    http.post<{ thread_id: string; created_at: string }>('/v1/conversations', { user_id: userId, title }),

  list: (userId: string, limit = 20, offset = 0) =>
    http.get<{ conversations: ConversationSummary[] }>('/v1/conversations', {
      params: { user_id: userId, limit, offset },
    }),

  getMessages: (threadId: string, userId: string, limit = 50, offset = 0) =>
    http.get<{ messages: Message[] }>(`/v1/conversations/${threadId}/messages`, {
      params: { user_id: userId, limit, offset },
    }),
}
