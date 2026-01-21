<script setup lang="ts">
import { NButton, NBadge } from 'naive-ui'
import { useApprovalStore } from '@/stores'
import ThemeToggle from '../common/ThemeToggle.vue'
import UserIdInput from '../common/UserIdInput.vue'

const emit = defineEmits<{
  'toggle-sidebar': []
}>()

const approvalStore = useApprovalStore()
</script>

<template>
  <header class="h-14 bg-panel border-b border-base flex items-center justify-between px-4">
    <!-- 左侧：菜单按钮 -->
    <div class="flex items-center gap-2">
      <NButton quaternary circle @click="emit('toggle-sidebar')">
        <template #icon>
          <span class="text-lg">☰</span>
        </template>
      </NButton>
      <span class="font-semibold text-lg hidden sm:inline">多智能体对话系统</span>
    </div>
    
    <!-- 右侧：工具栏 -->
    <div class="flex items-center gap-3">
      <!-- 用户 ID 输入 -->
      <UserIdInput />
      
      <!-- 人工审核按钮 -->
      <NBadge :value="approvalStore.pendingCount" :show="approvalStore.pendingCount > 0">
        <NButton @click="approvalStore.showPanel()">
          审核
        </NButton>
      </NBadge>
      
      <!-- 主题切换 -->
      <ThemeToggle />
    </div>
  </header>
</template>
