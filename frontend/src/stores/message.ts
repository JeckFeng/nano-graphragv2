import { defineStore } from 'pinia'
import { ref } from 'vue'
import { conversationApi } from '@/api/conversations'
import type { Message } from '@/types'

export const useMessageStore = defineStore('message', () => {
  const messages = ref<Message[]>([])
  const loading = ref(false)

  const loadMessages = async (threadId: string, userId: string) => {
    loading.value = true
    try {
      const res = await conversationApi.getMessages(threadId, userId)
      messages.value = res.data.messages
    } finally {
      loading.value = false
    }
  }

  const addMessage = (message: Message) => {
    messages.value.push(message)
  }

  const clearMessages = () => {
    messages.value = []
  }

  // 别名保持兼容
  const fetchMessages = loadMessages
  const clear = clearMessages

  return { messages, loading, loadMessages, fetchMessages, addMessage, clearMessages, clear }
})
