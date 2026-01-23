<script setup lang="ts">
import { onMounted, watch } from 'vue'
import { useMessage } from 'naive-ui'
import { useUserStore, useConversationStore, useMessageStore, useApprovalStore, useTraceStore } from '@/stores'
import { useToast } from '@/composables/useToast'
import LayoutShell from '@/components/layout/LayoutShell.vue'
import ChatView from '@/views/ChatView.vue'
import ApprovalPanel from '@/components/approval/ApprovalPanel.vue'

const userStore = useUserStore()
const conversationStore = useConversationStore()
const messageStore = useMessageStore()
const approvalStore = useApprovalStore()
const traceStore = useTraceStore()
const message = useMessage()

useToast()

const loadConversations = async () => {
  try {
    await conversationStore.fetchList(userStore.userId)
  } catch {
    message.error('加载会话列表失败')
  }
}

const loadApprovals = async () => {
  try {
    await approvalStore.fetchList(userStore.userId)
  } catch {
    // 静默失败
  }
}

const handleCreate = async () => {
  try {
    await conversationStore.create(userStore.userId)
    messageStore.clear()
    message.success('会话创建成功')
  } catch {
    message.error('创建会话失败')
  }
}

const handleSelect = (threadId: string) => {
  conversationStore.select(threadId)
}

watch(() => userStore.userId, () => {
  conversationStore.clear()
  messageStore.clear()
  traceStore.clearTraces()
  loadConversations()
  loadApprovals()
})

onMounted(() => {
  loadConversations()
  loadApprovals()
})
</script>

<template>
  <LayoutShell @create="handleCreate" @select="handleSelect">
    <ChatView />
  </LayoutShell>
  <ApprovalPanel />
</template>
