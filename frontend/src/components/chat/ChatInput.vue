<script setup lang="ts">
import { ref } from 'vue'
import { NInput, NButton } from 'naive-ui'

defineProps<{
  disabled: boolean
}>()

const emit = defineEmits<{
  send: [content: string]
}>()

const inputValue = ref('')
const sending = ref(false)

const handleSend = () => {
  if (sending.value) return
  const content = inputValue.value.trim()
  if (content) {
    sending.value = true
    emit('send', content)
    inputValue.value = ''
    // 防抖：500ms 后允许再次发送
    setTimeout(() => { sending.value = false }, 500)
  }
}

const handleKeydown = (e: KeyboardEvent) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}
</script>

<template>
  <div class="border-t border-base p-4 bg-panel">
    <div class="flex gap-3">
      <NInput
        v-model:value="inputValue"
        type="textarea"
        placeholder="输入消息，按 Enter 发送..."
        :autosize="{ minRows: 1, maxRows: 4 }"
        :disabled="disabled"
        @keydown="handleKeydown"
      />
      <NButton
        type="primary"
        :disabled="disabled || !inputValue.trim() || sending"
        :loading="sending"
        @click="handleSend"
      >
        发送
      </NButton>
    </div>
  </div>
</template>
