import { defineStore } from 'pinia'
import { ref } from 'vue'
import { approvalApi } from '@/api/approvals'
import type { ApprovalRecord } from '@/types'

export const useApprovalStore = defineStore('approval', () => {
  const list = ref<ApprovalRecord[]>([])
  const pendingCount = ref(0)
  const panelVisible = ref(false)
  const selectedId = ref<string | null>(null)
  const resolvingIds = ref<Set<string>>(new Set())
  const pollingIds = ref<Set<string>>(new Set())

  const isResolving = (approvalId: string) => resolvingIds.value.has(approvalId)

  const fetchList = async (userId: string) => {
    const res = await approvalApi.list(userId, 'pending')
    list.value = res.data.approvals
    pendingCount.value = list.value.length
  }

  const resolve = async (userId: string, approvalId: string, decision: string, editedArgs?: Record<string, unknown>) => {
    if (resolvingIds.value.has(approvalId)) {
      throw new Error('该审批正在处理中')
    }
    
    resolvingIds.value.add(approvalId)
    try {
      const res = await approvalApi.resolve(approvalId, userId, decision, editedArgs)
      
      // 异步模式：status 为 processing 时，从列表移除，等待 WS 推送结果
      if (res.data.status === 'processing') {
        list.value = list.value.filter((a) => a.approval_id !== approvalId)
        pendingCount.value = list.value.length
        return { ...res.data, isAsync: true }
      }
      
      // 同步完成（已处理过的审批）
      list.value = list.value.filter((a) => a.approval_id !== approvalId)
      pendingCount.value = list.value.length
      return res.data
    } finally {
      resolvingIds.value.delete(approvalId)
    }
  }

  // 轮询审批状态（WebSocket 断开时的降级方案）
  const pollStatus = async (
    userId: string,
    approvalId: string,
    onComplete: (content: string | null) => void,
    maxAttempts = 60,
    interval = 5000
  ) => {
    if (pollingIds.value.has(approvalId)) return
    
    pollingIds.value.add(approvalId)
    try {
      for (let i = 0; i < maxAttempts; i++) {
        const res = await approvalApi.getStatus(approvalId, userId)
        if (res.data.status !== 'processing') {
          onComplete(res.data.result_content)
          return
        }
        await new Promise(resolve => setTimeout(resolve, interval))
      }
      // 超时
      onComplete(null)
    } finally {
      pollingIds.value.delete(approvalId)
    }
  }

  const showPanel = () => { panelVisible.value = true }
  const hidePanel = () => { panelVisible.value = false }
  const select = (id: string) => { selectedId.value = id }

  return { list, pendingCount, panelVisible, selectedId, resolvingIds, pollingIds, isResolving, fetchList, resolve, pollStatus, showPanel, hidePanel, select }
})
