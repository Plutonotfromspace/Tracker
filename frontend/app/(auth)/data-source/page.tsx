"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Database, Sparkles } from "lucide-react"

export default function DataSourcePage() {
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    // Check authentication
    const token = localStorage.getItem("auth_token")
    if (!token) {
      router.push("/login")
    }
  }, [router])

  const handleSelection = (source: "live" | "mock") => {
    setIsLoading(true)
    localStorage.setItem("data_source", source)
    router.push("/jobs")
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900 p-4">
      <div className="w-full max-w-4xl">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">Choose Your Data Source</h1>
          <p className="text-muted-foreground">
            Select how you'd like to view the dashboard
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Live Data Option */}
          <Card className="relative hover:shadow-lg transition-shadow cursor-pointer border-2 hover:border-primary">
            <CardHeader>
              <div className="flex items-center gap-3 mb-2">
                <div className="rounded-full bg-primary/10 p-2">
                  <Database className="h-6 w-6 text-primary" />
                </div>
                <CardTitle className="text-xl">Live Data</CardTitle>
              </div>
              <CardDescription>
                Connect to the real-time database
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                View actual truck detection data from your video processing jobs. 
                This is the production environment where you'll see real results, 
                classifications, and analytics.
              </p>
              <ul className="text-sm space-y-2 text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="text-primary mt-0.5">•</span>
                  <span>Real-time job processing results</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary mt-0.5">•</span>
                  <span>Actual truck detections and classifications</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary mt-0.5">•</span>
                  <span>Empty if no videos have been processed yet</span>
                </li>
              </ul>
              <Button 
                onClick={() => handleSelection("live")}
                disabled={isLoading}
                className="w-full"
              >
                Use Live Data
              </Button>
            </CardContent>
          </Card>

          {/* Mock Data Option */}
          <Card className="relative hover:shadow-lg transition-shadow cursor-pointer border-2 hover:border-primary">
            <CardHeader>
              <div className="flex items-center gap-3 mb-2">
                <div className="rounded-full bg-purple-500/10 p-2">
                  <Sparkles className="h-6 w-6 text-purple-500" />
                </div>
                <CardTitle className="text-xl">Mock Data</CardTitle>
              </div>
              <CardDescription>
                Explore with sample data
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Experience the dashboard with pre-populated sample data. 
                Perfect for exploring features, understanding the interface, 
                and seeing what the system looks like with data.
              </p>
              <ul className="text-sm space-y-2 text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="text-purple-500 mt-0.5">•</span>
                  <span>Sample jobs and truck detections</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-500 mt-0.5">•</span>
                  <span>Pre-classified trucks with example data</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-500 mt-0.5">•</span>
                  <span>Ideal for demo and testing purposes</span>
                </li>
              </ul>
              <Button 
                onClick={() => handleSelection("mock")}
                disabled={isLoading}
                variant="secondary"
                className="w-full"
              >
                Use Mock Data
              </Button>
            </CardContent>
          </Card>
        </div>

        <p className="text-center text-sm text-muted-foreground mt-6">
          You can change this setting anytime from your dashboard
        </p>
      </div>
    </div>
  )
}
