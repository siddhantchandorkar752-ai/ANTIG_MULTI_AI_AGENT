import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'OMEGA Research Grid — Autonomous Multi-Agent Research OS',
  description: 'Production-grade autonomous AI research platform powered by multi-agent collaboration, real-time search, and intelligent synthesis.',
  keywords: ['AI research', 'multi-agent', 'autonomous AI', 'research assistant', 'LangGraph'],
  authors: [{ name: 'OMEGA Research Grid' }],
  openGraph: {
    title: 'OMEGA Research Grid',
    description: 'Autonomous Multi-Agent Research Operating System',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-omega-bg text-omega-text antialiased`}>
        {children}
      </body>
    </html>
  )
}
