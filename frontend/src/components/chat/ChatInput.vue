<script setup lang="ts">
import { ref } from 'vue'
import { NInput, NButton } from 'naive-ui'

const props = defineProps<{
  disabled: boolean
}>()

const emit = defineEmits<{
  send: [content: string]
}>()

const inputValue = ref('')

const handleSend = () => {
  if (props.disabled) return
  const content = inputValue.value.trim()
  if (content) {
    emit('send', content)
    inputValue.value = ''
  }
}

const handleKeydown = (e: KeyboardEvent) => {
  if (e.key === 'Enter' && !e.shiftKey && !props.disabled) {
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
        :placeholder="disabled ? '请等待响应...' : '输入消息，按 Enter 发送...'"
        :autosize="{ minRows: 1, maxRows: 4 }"
        :disabled="disabled"
        @keydown="handleKeydown"
      />
      <NButton
        type="primary"
        :disabled="disabled || !inputValue.trim()"
        @click="handleSend"
      >
        {{ disabled ? '处理中' : '发送' }}
      </NButton>
    </div>
  </div>
</template>
