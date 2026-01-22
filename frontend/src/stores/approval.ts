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
      list.value = list.value.filter((a) => a.approval_id !== approvalId)
      if (res.data.next_approval) {
        list.value.unshift(res.data.next_approval)
      }
      pendingCount.value = list.value.length
      return res.data
    } finally {
      resolvingIds.value.delete(approvalId)
    }
  }

  const showPanel = () => { panelVisible.value = true }
  const hidePanel = () => { panelVisible.value = false }
  const select = (id: string) => { selectedId.value = id }

  return { list, pendingCount, panelVisible, selectedId, resolvingIds, isResolving, fetchList, resolve, showPanel, hidePanel, select }
})
