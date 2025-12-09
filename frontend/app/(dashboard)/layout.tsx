"use client"

import { AppSidebar } from "@/components/app-sidebar";
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { LogOut, Database } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [username, setUsername] = useState("");
  const [dataSource, setDataSource] = useState<"live" | "mock">("live");
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [showSpinner, setShowSpinner] = useState(false);
  const [fadeOutSpinner, setFadeOutSpinner] = useState(false);
  const [fadeIn, setFadeIn] = useState(true);

  useEffect(() => {
    // Check authentication
    const token = localStorage.getItem("auth_token");
    const user = localStorage.getItem("username");
    const dataSourceValue = localStorage.getItem("data_source");
    
    if (!token) {
      router.push("/login");
    } else if (!dataSourceValue) {
      router.push("/data-source");
    } else {
      setIsAuthenticated(true);
      setUsername(user || "User");
      setDataSource(dataSourceValue as "live" | "mock");
    }
    setIsLoading(false);
  }, [router]);

  const handleDataSourceToggle = (checked: boolean) => {
    const newSource = checked ? "mock" : "live";
    
    // Start the slide out animation
    setIsTransitioning(true);
    setFadeIn(false);
    
    // After slide out completes, show spinner and update data
    setTimeout(() => {
      setShowSpinner(true);
      setDataSource(newSource);
      localStorage.setItem("data_source", newSource);
      
      // Minimum delay to show spinner (prevents jarring instant loads)
      const minDelay = 800;
      const startTime = Date.now();
      
      // Wait for minimum delay, then fade out and reload/redirect
      const elapsed = Date.now() - startTime;
      const remainingDelay = Math.max(0, minDelay - elapsed);
      
      setTimeout(() => {
        setFadeOutSpinner(true);
        setTimeout(() => {
          // If on a specific job detail page, redirect to jobs list then reload
          if (window.location.pathname.includes("/jobs/") && window.location.pathname !== "/jobs") {
            window.location.href = "/jobs";
          } else {
            window.location.reload();
          }
        }, 300); // Match fade out duration
      }, remainingDelay);
    }, 500); // After slide out animation
  };

  const handleLogout = () => {
    localStorage.removeItem("auth_token");
    localStorage.removeItem("username");
    localStorage.removeItem("data_source");
    router.push("/login");
  };

  if (isLoading || !isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
        </div>
      </div>
    );
  }

  return (
    <>
      <div 
        className={`transition-opacity duration-500 ease-in-out ${
          isTransitioning ? "opacity-0" : "opacity-100"
        } ${fadeIn ? "animate-in fade-in duration-700" : ""}`}
      >
        <SidebarProvider>
          <AppSidebar />
          <SidebarInset>
            <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
              <SidebarTrigger className="-ml-1" />
              <div className="ml-auto flex items-center gap-4">
                <div className="flex items-center gap-3 px-3 py-1.5 rounded-md border bg-muted/30">
                  <Database className="h-4 w-4" style={{ color: dataSource === "live" ? "hsl(142, 71%, 45%)" : "hsl(271, 91%, 65%)" }} />
                  <Label htmlFor="data-source-switch" className="text-sm font-medium cursor-pointer"
                         style={{ color: dataSource === "live" ? "hsl(142, 71%, 35%)" : "hsl(271, 91%, 45%)" }}>
                    {dataSource === "live" ? "Live Data" : "Mock Data"}
                  </Label>
                  <Switch 
                    id="data-source-switch"
                    checked={dataSource === "mock"}
                    onCheckedChange={handleDataSourceToggle}
                    style={{
                      backgroundColor: dataSource === "live" ? "hsl(142, 71%, 45%)" : "hsl(271, 91%, 65%)"
                    } as any}
                  />
                </div>
                <Separator orientation="vertical" className="h-6" />
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleLogout}
                  className="gap-2"
                >
                  <LogOut className="h-4 w-4" />
                  Logout
                </Button>
              </div>
            </header>
            <main className="flex-1 p-6">
              {children}
            </main>
          </SidebarInset>
        </SidebarProvider>
      </div>

      {/* Spinner overlay - shows when transitioning */}
      {showSpinner && (
        <div className={`fixed inset-0 z-50 flex items-center justify-center bg-background transition-opacity duration-300 ${
          fadeOutSpinner ? "opacity-0" : "opacity-100"
        }`}>
          <div className="text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary mx-auto"></div>
            <p className="mt-4 text-muted-foreground">Switching data source...</p>
          </div>
        </div>
      )}
    </>
  );
}
