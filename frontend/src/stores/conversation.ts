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
    // 生成带时间戳的默认标题
    const now = new Date()
    const timestamp = now.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    }).replace(/\//g, '-')
    const defaultTitle = title || `新会话[${timestamp}]`
    
    const res = await conversationApi.create(userId, defaultTitle)
    const newConv: ConversationSummary = {
      thread_id: res.data.thread_id,
      title: defaultTitle,
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

  const remove = async (userId: string, threadId: string) => {
    await conversationApi.delete(threadId, userId)
    list.value = list.value.filter((c) => c.thread_id !== threadId)
    if (currentId.value === threadId) {
      currentId.value = list.value.length > 0 ? list.value[0].thread_id : null
    }
  }

  return { list, currentId, currentThreadId, loading, fetchList, create, select, clear, remove }
})
