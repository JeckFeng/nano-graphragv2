import { defineStore } from 'pinia'
import { ref } from 'vue'
import { approvalApi } from '@/api/approvals'
import type { ApprovalRecord } from '@/types'

export const useApprovalStore = defineStore('approval', () => {
  const list = ref<ApprovalRecord[]>([])
  const pendingCount = ref(0)
  const panelVisible = ref(false)
  const selectedId = ref<string | null>(null)

  const fetchList = async (userId: string) => {
    const res = await approvalApi.list(userId, 'pending')
    list.value = res.data.approvals
    pendingCount.value = list.value.length
  }

  const resolve = async (userId: string, approvalId: string, decision: string, editedArgs?: Record<string, unknown>) => {
    const res = await approvalApi.resolve(approvalId, userId, decision, editedArgs)
    list.value = list.value.filter((a) => a.approval_id !== approvalId)
    if (res.data.next_approval) {
      list.value.unshift(res.data.next_approval)
    }
    pendingCount.value = list.value.length
    return res.data
  }

  const showPanel = () => { panelVisible.value = true }
  const hidePanel = () => { panelVisible.value = false }
  const select = (id: string) => { selectedId.value = id }

  return { list, pendingCount, panelVisible, selectedId, fetchList, resolve, showPanel, hidePanel, select }
})
