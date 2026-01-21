import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { conversationApi } from '@/api/conversations'
import type { ConversationSummary } from '@/types'

export const useConversationStore = defineStore('conversation', () => {
  const list = ref<ConversationSummary[]>([])
  const currentId = ref<string | null>(null)
  const loading = ref(false)

  // 别名
  const currentThreadId = computed(() => currentId.value)

  const fetchList = async (userId: string) => {
    loading.value = true
    try {
      const res = await conversationApi.list(userId)
      list.value = res.data.conversations
    } finally {
      loading.value = false
    }
  }

  const create = async (userId: string, title?: string) => {
    const res = await conversationApi.create(userId, title)
    const newConv: ConversationSummary = {
      thread_id: res.data.thread_id,
      title: title || null,
      created_at: res.data.created_at,
    }
    list.value.unshift(newConv)
    currentId.value = newConv.thread_id
    return newConv
  }

  const select = (threadId: string) => {
    currentId.value = threadId
  }

  const clear = () => {
    currentId.value = null
  }

  return { list, currentId, currentThreadId, loading, fetchList, create, select, clear }
})
