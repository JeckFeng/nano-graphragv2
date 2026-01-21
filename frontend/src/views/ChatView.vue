<script setup lang="ts">
import { watch, onMounted } from 'vue'
import { NEmpty } from 'naive-ui'
import { useConversationStore, useMessageStore, useUserStore, useApprovalStore } from '@/stores'
import ChatPanel from '@/components/chat/ChatPanel.vue'

const conversationStore = useConversationStore()
const messageStore = useMessageStore()
const userStore = useUserStore()
const approvalStore = useApprovalStore()

const handleApprovalRequired = async (approvalId: string) => {
  // 刷新审批列表并打开抽屉
  await approvalStore.fetchList(userStore.userId)
  approvalStore.select(approvalId)
  approvalStore.showPanel()
}

// 切换会话时加载消息
watch(
  () => conversationStore.currentThreadId,
  async (threadId) => {
    if (threadId) {
      await messageStore.loadMessages(threadId, userStore.userId)
    } else {
      messageStore.clearMessages()
    }
  }
)

onMounted(() => {
  if (conversationStore.currentThreadId) {
    messageStore.loadMessages(conversationStore.currentThreadId, userStore.userId)
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
