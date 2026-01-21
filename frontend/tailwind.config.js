/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{vue,js,ts,jsx,tsx}',
  ],
  darkMode: ['class', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        primary: 'var(--primary)',
        accent: 'var(--accent)',
        muted: 'var(--muted)',
      },
      backgroundColor: {
        base: 'var(--bg)',
        panel: 'var(--panel)',
        'bubble-user': 'var(--bubble-user)',
        'bubble-ai': 'var(--bubble-ai)',
      },
      textColor: {
        base: 'var(--text)',
        muted: 'var(--muted)',
      },
      borderColor: {
        base: 'var(--border)',
      },
    },
  },
  plugins: [],
}
