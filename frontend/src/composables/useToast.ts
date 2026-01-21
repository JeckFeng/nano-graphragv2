import { useMessage } from 'naive-ui'
import { onMounted, onUnmounted } from 'vue'

export function useToast() {
  const message = useMessage()

  const handleApiError = (e: Event) => {
    const detail = (e as CustomEvent).detail
    message.error(detail || '请求失败')
  }

  onMounted(() => {
    window.addEventListener('api-error', handleApiError)
  })

  onUnmounted(() => {
    window.removeEventListener('api-error', handleApiError)
  })

  return { message }
}
