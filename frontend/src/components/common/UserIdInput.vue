<script setup lang="ts">
import { ref, watch } from 'vue'
import { NInput } from 'naive-ui'
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()
const draftUserId = ref(userStore.userId)

watch(
  () => userStore.userId,
  (value) => {
    draftUserId.value = value
  }
)

const commitUserId = () => {
  userStore.setUserId(draftUserId.value)
}

const handleKeydown = (e: KeyboardEvent) => {
  if (e.key === 'Enter') {
    e.preventDefault()
    commitUserId()
  }
}
</script>

<template>
  <div class="flex items-center gap-2">
    <span class="text-sm text-muted hidden md:inline">用户ID:</span>
    <NInput
      v-model:value="draftUserId"
      placeholder="输入用户ID"
      size="small"
      style="width: 100px"
      @blur="commitUserId"
      @keydown="handleKeydown"
    />
  </div>
</template>
