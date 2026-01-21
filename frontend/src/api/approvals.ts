import { http } from './index'
import type { ApprovalRecord, ApprovalResolutionResponse } from '@/types'

export const approvalApi = {
  list: (userId: string, status?: string, threadId?: string, limit = 20, offset = 0) =>
    http.get<{ approvals: ApprovalRecord[] }>('/v1/approvals', {
      params: { user_id: userId, status, thread_id: threadId, limit, offset },
    }),

  get: (approvalId: string, userId: string) =>
    http.get<ApprovalRecord>(`/v1/approvals/${approvalId}`, {
      params: { user_id: userId },
    }),

  resolve: (approvalId: string, userId: string, decision: string, editedArgs?: Record<string, unknown>) =>
    http.post<ApprovalResolutionResponse>(`/v1/approvals/${approvalId}`, {
      decision,
      edited_args: editedArgs,
    }, { params: { user_id: userId } }),
}
