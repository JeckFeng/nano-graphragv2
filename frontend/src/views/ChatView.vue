<script setup lang="ts">
import { watch, onMounted } from 'vue'
import { NEmpty, useMessage } from 'naive-ui'
import { useConversationStore, useMessageStore, useUserStore, useApprovalStore, useTraceStore } from '@/stores'
import ChatPanel from '@/components/chat/ChatPanel.vue'

const conversationStore = useConversationStore()
const messageStore = useMessageStore()
const userStore = useUserStore()
const approvalStore = useApprovalStore()
const traceStore = useTraceStore()
const message = useMessage()
const traceEnabled = String(import.meta.env.VITE_TRACE_PANEL_ENABLED).toLowerCase() === 'true'

// 检查当前会话是否有待处理的审批
const checkPendingApprovals = async () => {
  if (!conversationStore.currentThreadId) return
  
  try {
    await approvalStore.fetchList(userStore.userId)
    const pendingForThread = approvalStore.list.find(
      a => a.thread_id === conversationStore.currentThreadId && a.status === 'pending'
    )
    if (pendingForThread) {
      message.warning('检测到未完成的审批，请先处理')
      approvalStore.select(pendingForThread.approval_id)
      approvalStore.showPanel()
    }
  } catch {
    // 静默失败
  }
}

const handleApprovalRequired = async (approvalId: string) => {
  await approvalStore.fetchList(userStore.userId)
  approvalStore.select(approvalId)
  approvalStore.showPanel()
}

// 切换会话时加载消息并检查审批
watch(
  () => conversationStore.currentThreadId,
  async (threadId) => {
    if (threadId) {
      await messageStore.loadMessages(threadId, userStore.userId)
      if (traceEnabled) {
        await traceStore.loadThreadTraces(threadId, userStore.userId)
      }
      await checkPendingApprovals()
    } else {
      messageStore.clearMessages()
      traceStore.clearTraces()
    }
  }
)

onMounted(async () => {
  if (conversationStore.currentThreadId) {
    await messageStore.loadMessages(conversationStore.currentThreadId, userStore.userId)
    if (traceEnabled) {
      await traceStore.loadThreadTraces(conversationStore.currentThreadId, userStore.userId)
    }
    await checkPendingApprovals()
  }
})
</script>

<template>
  <div class="h-full">
    <template v-if="conversationStore.currentThreadId">
      <ChatPanel
        :thread-id="conversationStore.currentThreadId"
        @approval-required="handleApprovalRequired"
      />
    </template>
    
    <div v-else class="h-full flex items-center justify-center">
      <NEmpty description="选择或创建一个会话开始对话" />
    </div>
  </div>
</template>
