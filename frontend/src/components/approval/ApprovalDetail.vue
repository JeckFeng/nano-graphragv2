<script setup lang="ts">
import { ref, computed } from 'vue'
import { NButton, NSpace, NTag, NDivider, useMessage } from 'naive-ui'
import { useApprovalStore, useUserStore, useConversationStore, useMessageStore } from '@/stores'
import type { ApprovalRecord } from '@/types'
import JsonEditor from './JsonEditor.vue'

const props = defineProps<{
  approval: ApprovalRecord
}>()

const emit = defineEmits<{
  resolved: []
}>()

const approvalStore = useApprovalStore()
const userStore = useUserStore()
const conversationStore = useConversationStore()
const messageStore = useMessageStore()
const message = useMessage()

const submitting = ref(false)
const isEditing = ref(false)
const editedArgs = ref<Record<string, unknown>>({})

// 提取第一个工具调用信息
const toolInfo = computed(() => {
  const interrupt = props.approval.interrupts[0]
  if (!interrupt?.action_requests[0]) return null
  return interrupt.action_requests[0]
})

const allowedDecisions = computed(() => {
  const interrupt = props.approval.interrupts[0]
  return interrupt?.review_configs[0]?.allowed_decisions || ['approve', 'reject']
})

const canEdit = computed(() => allowedDecisions.value.includes('edit'))

const startEdit = () => {
  if (toolInfo.value) {
    editedArgs.value = { ...(toolInfo.value.args ?? {}) }
    isEditing.value = true
  }
}

const cancelEdit = () => {
  isEditing.value = false
}

const handleDecision = async (decision: string) => {
  try {
    submitting.value = true
    const args = decision === 'edit' ? editedArgs.value : undefined
    const result = await approvalStore.resolve(userStore.userId, props.approval.approval_id, decision, args)
    if (result.result_content && props.approval.thread_id === conversationStore.currentThreadId) {
      messageStore.addMessage({
        id: Date.now(),
        role: 'assistant',
        content: result.result_content,
        created_at: new Date().toISOString(),
      })
    }
    message.success(decision === 'approve' ? '已批准' : decision === 'reject' ? '已拒绝' : '已编辑并批准')
    isEditing.value = false
    emit('resolved')
  } catch {
    message.error('提交决策失败')
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <div class="p-4">
    <div class="flex items-center gap-2 mb-4">
      <NTag :type="approval.status === 'pending' ? 'warning' : 'success'">
        {{ approval.status }}
      </NTag>
      <span class="text-sm text-muted">{{ approval.created_at }}</span>
    </div>

    <template v-if="toolInfo">
      <div class="mb-2 font-medium">工具: {{ toolInfo.name }}</div>
      
      <NDivider />
      
      <div class="mb-2 text-sm text-muted">参数:</div>
      
      <template v-if="isEditing">
        <JsonEditor v-model="editedArgs" />
        <NSpace class="mt-4">
          <NButton type="primary" :loading="submitting" @click="handleDecision('edit')">
            保存并批准
          </NButton>
          <NButton @click="cancelEdit">取消</NButton>
        </NSpace>
      </template>
      
      <template v-else>
        <pre class="bg-base p-3 rounded text-sm overflow-auto">{{ JSON.stringify(toolInfo.args, null, 2) }}</pre>
        
        <NSpace class="mt-4" v-if="approval.status === 'pending'">
          <NButton type="success" :loading="submitting" @click="handleDecision('approve')">
            批准
          </NButton>
          <NButton type="error" :loading="submitting" @click="handleDecision('reject')">
            拒绝
          </NButton>
          <NButton v-if="canEdit" @click="startEdit">
            编辑
          </NButton>
        </NSpace>
      </template>
    </template>
  </div>
</template>
