<script setup lang="ts">
import { ref, watch } from 'vue'
import { NInput, NAlert } from 'naive-ui'

const props = defineProps<{
  modelValue: Record<string, unknown>
}>()

const emit = defineEmits<{
  'update:modelValue': [value: Record<string, unknown>]
}>()

const jsonText = ref('')
const error = ref('')

watch(
  () => props.modelValue,
  (val) => {
    jsonText.value = JSON.stringify(val, null, 2)
    error.value = ''
  },
  { immediate: true }
)

const handleInput = (val: string) => {
  jsonText.value = val
  try {
    const parsed = JSON.parse(val)
    error.value = ''
    emit('update:modelValue', parsed)
  } catch {
    error.value = 'JSON 格式错误'
  }
}
</script>

<template>
  <div>
    <NInput
      type="textarea"
      :value="jsonText"
      :autosize="{ minRows: 4, maxRows: 12 }"
      font-family="monospace"
      @update:value="handleInput"
    />
    <NAlert v-if="error" type="error" class="mt-2" :show-icon="false">
      {{ error }}
    </NAlert>
  </div>
</template>
