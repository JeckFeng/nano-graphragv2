<script setup lang="ts">
import { computed, watch } from 'vue'
import { NDrawer, NDrawerContent } from 'naive-ui'
import { useApprovalStore, useUserStore } from '@/stores'
import ApprovalList from './ApprovalList.vue'
import ApprovalDetail from './ApprovalDetail.vue'

const approvalStore = useApprovalStore()
const userStore = useUserStore()

const selectedApproval = computed(() =>
  approvalStore.list.find((a) => a.approval_id === approvalStore.selectedId)
)

const handleSelect = (id: string) => {
  approvalStore.select(id)
}

const handleResolved = () => {
  // 自动选择下一个待审批项
  if (approvalStore.list.length > 0) {
    approvalStore.select(approvalStore.list[0].approval_id)
  } else {
    approvalStore.hidePanel()
  }
}

// 打开面板时加载列表
watch(
  () => approvalStore.panelVisible,
  (visible) => {
    if (visible) {
      approvalStore.fetchList(userStore.userId)
    }
  }
)
</script>

<template>
  <NDrawer
    :show="approvalStore.panelVisible"
    :width="600"
    placement="right"
    @update:show="(v) => v ? approvalStore.showPanel() : approvalStore.hidePanel()"
  >
    <NDrawerContent title="待审批列表" closable>
      <div class="flex h-full">
        <!-- 左侧列表 -->
        <div class="w-48 border-r border-base">
          <ApprovalList
            :approvals="approvalStore.list"
            :selected-id="approvalStore.selectedId"
            @select="handleSelect"
          />
        </div>
        
        <!-- 右侧详情 -->
        <div class="flex-1">
          <ApprovalDetail
            v-if="selectedApproval"
            :approval="selectedApproval"
            @resolved="handleResolved"
          />
          <div v-else class="h-full flex items-center justify-center text-muted">
            选择一个审批项查看详情
          </div>
        </div>
      </div>
    </NDrawerContent>
  </NDrawer>
</template>
