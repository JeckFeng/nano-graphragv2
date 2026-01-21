import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

const STORAGE_KEY = 'user_id'
const DEFAULT_USER_ID = '001'

export const useUserStore = defineStore('user', () => {
  const userId = ref(localStorage.getItem(STORAGE_KEY) || DEFAULT_USER_ID)

  watch(userId, (val) => {
    localStorage.setItem(STORAGE_KEY, val)
  })

  const setUserId = (id: string) => {
    userId.value = id.trim() || DEFAULT_USER_ID
  }

  return { userId, setUserId }
})
