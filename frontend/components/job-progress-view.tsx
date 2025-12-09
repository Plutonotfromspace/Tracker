"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Loader2, Clock, CheckCircle2, ArrowLeft, CheckCircle } from "lucide-react";
import Link from "next/link";
import { Job } from "@/lib/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ProgressData {
  job_id: string;
  status: string;
  current_stage: string | null;
  progress_percent: number;
  current_frame: number;
  total_frames: number;
  eta_seconds: number | null;
  stage_eta_seconds: number | null;
}

interface JobProgressViewProps {
  initialJob: Job;
}

export function JobProgressView({ initialJob }: JobProgressViewProps) {
  const router = useRouter();
  const [job, setJob] = useState<Job>(initialJob);
  const [progress, setProgress] = useState<ProgressData | null>(null);
  const [isPolling, setIsPolling] = useState(true);

  // Fetch initial progress data immediately
  useEffect(() => {
    const fetchInitialProgress = async () => {
      try {
        const token = localStorage.getItem("auth_token");
        const progressResponse = await fetch(`${API_URL}/api/jobs/${job.id}/progress`, {
          headers: { Authorization: `Bearer ${token}` },
        });

        if (progressResponse.ok) {
          const progressData: ProgressData = await progressResponse.json();
          setProgress(progressData);
        }
      } catch (error) {
        console.error("Failed to fetch initial progress:", error);
      }
    };

    fetchInitialProgress();
  }, [job.id]);

  useEffect(() => {
    // Poll for progress updates while job is active
    // Note: Even demo users need real-time updates when they upload actual videos
    if (!isPolling || job.status === "completed" || job.status === "failed") {
      return;
    }

    const pollInterval = setInterval(async () => {
      try {
        const token = localStorage.getItem("auth_token");
        
        // Poll progress endpoint for real-time updates
        const progressResponse = await fetch(`${API_URL}/api/jobs/${job.id}/progress`, {
          headers: { Authorization: `Bearer ${token}` },
        });

        if (progressResponse.ok) {
          const progressData: ProgressData = await progressResponse.json();
          setProgress(progressData);
          
          // Update job status if changed
          if (progressData.status !== job.status) {
            setJob({ ...job, status: progressData.status as Job["status"] });
          }

          // If completed, redirect to full view
          if (progressData.status === "completed") {
            setIsPolling(false);
            setTimeout(() => {
              // Force navigation to reload the page with completed status
              window.location.reload();
            }, 1000);
          }
        }
      } catch (error) {
        console.error("Failed to poll job progress:", error);
      }
    }, 2000); // Poll every 2 seconds for responsive updates

    return () => clearInterval(pollInterval);
  }, [job, isPolling, router, setJob]);

  const getStatusMessage = () => {
    switch (job.status) {
      case "created":
        return {
          title: "Job Queued",
          message: "Your job is waiting in the queue and will start processing shortly.",
          icon: <Clock className="h-12 w-12 text-muted-foreground" />,
        };
      case "processing":
        return {
          title: "Processing Video",
          message: "Analyzing video and detecting vehicles. This may take several minutes depending on video length.",
          icon: <Loader2 className="h-12 w-12 text-primary animate-spin" />,
        };
      case "completed":
        return {
          title: "Processing Complete",
          message: "Your job has finished processing. Redirecting to results...",
          icon: <CheckCircle2 className="h-12 w-12 text-green-600" />,
        };
      default:
        return {
          title: "Unknown Status",
          message: "Processing status unknown.",
          icon: <Clock className="h-12 w-12 text-muted-foreground" />,
        };
    }
  };

  const statusInfo = getStatusMessage();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <Button variant="ghost" size="sm" asChild className="mb-4">
          <Link href="/jobs">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Jobs
          </Link>
        </Button>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Job Status</h1>
            <p className="text-muted-foreground mt-1 font-mono text-sm">
              {job.id}
            </p>
          </div>
          <Badge 
            variant={job.status === "created" ? "secondary" : "default"}
            className={job.status === "processing" ? "animate-pulse bg-blue-600 hover:bg-blue-600" : ""}
          >
            {job.status === "created" ? "Queued" : "Processing"}
          </Badge>
        </div>
      </div>

      {/* Status Card */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col items-center justify-center py-12 space-y-6">
            {/* Stage Icon */}
            {progress?.current_stage === "loading" && (
              <Loader2 className="h-16 w-16 animate-spin text-blue-600" />
            )}
            {progress?.current_stage === "detecting" && (
              <Loader2 className="h-16 w-16 animate-spin text-blue-600" />
            )}
            {progress?.current_stage === "classifying" && (
              <Loader2 className="h-16 w-16 animate-spin text-blue-600" />
            )}
            {progress?.current_stage === "analyzing" && (
              <Loader2 className="h-16 w-16 animate-spin text-blue-600" />
            )}
            {!progress?.current_stage && statusInfo.icon}

            {/* Stage Title and Description */}
            <div className="text-center space-y-2">
              {progress?.current_stage && (
                <>
                  <h2 className="text-2xl font-semibold capitalize">
                    {progress.current_stage}
                  </h2>
                  <p className="text-muted-foreground max-w-md">
                    {progress.current_stage === "loading" && "Preparing video for analysis"}
                    {progress.current_stage === "detecting" && "Analyzing video and detecting vehicles"}
                    {progress.current_stage === "classifying" && "Classifying detected vehicles"}
                    {progress.current_stage === "analyzing" && "Generating summary reports"}
                  </p>
                </>
              )}
              {!progress?.current_stage && (
                <>
                  <h2 className="text-2xl font-semibold">{statusInfo.title}</h2>
                  <p className="text-muted-foreground max-w-md">
                    {statusInfo.message}
                  </p>
                </>
              )}
            </div>

            {/* Progress Bar */}
            {progress && (
              <div className="w-full max-w-2xl space-y-4">
                <Progress value={progress.progress_percent} className="h-2" />
                
                {/* Percentage Display */}
                <div className="text-center">
                  <p className="text-3xl font-bold">{progress.progress_percent.toFixed(1)}%</p>
                </div>

                {/* Stage Badges */}
                <div className="flex items-center justify-center gap-2 flex-wrap">
                  {/* Stage 1: Loading */}
                  <Badge 
                    variant={progress.current_stage === "loading" ? "default" : "secondary"}
                    className="px-3 py-1.5 text-sm"
                  >
                    {progress.current_stage === "loading" && <Loader2 className="h-3 w-3 animate-spin mr-1.5" />}
                    {progress.current_stage !== "loading" && <CheckCircle className="h-3 w-3 mr-1.5" />}
                    1. Loading
                  </Badge>

                  <div className="h-px w-4 bg-border" />

                  {/* Stage 2: Detecting */}
                  <Badge 
                    variant={progress.current_stage === "detecting" ? "default" : progress.current_stage === "loading" ? "outline" : "secondary"}
                    className="px-3 py-1.5 text-sm"
                  >
                    {progress.current_stage === "detecting" && <Loader2 className="h-3 w-3 animate-spin mr-1.5" />}
                    {progress.current_stage !== "detecting" && progress.current_stage !== "loading" && <CheckCircle className="h-3 w-3 mr-1.5" />}
                    2. Detecting
                  </Badge>

                  <div className="h-px w-4 bg-border" />

                  {/* Stage 3: Classifying */}
                  <Badge 
                    variant={progress.current_stage === "classifying" ? "default" : progress.current_stage === "analyzing" ? "secondary" : "outline"}
                    className="px-3 py-1.5 text-sm"
                  >
                    {progress.current_stage === "classifying" && <Loader2 className="h-3 w-3 animate-spin mr-1.5" />}
                    {progress.current_stage === "analyzing" && <CheckCircle className="h-3 w-3 mr-1.5" />}
                    3. Classifying
                  </Badge>

                  <div className="h-px w-4 bg-border" />

                  {/* Stage 4: Analyzing */}
                  <Badge 
                    variant={progress.current_stage === "analyzing" ? "default" : "outline"}
                    className="px-3 py-1.5 text-sm"
                  >
                    {progress.current_stage === "analyzing" && <Loader2 className="h-3 w-3 animate-spin mr-1.5" />}
                    4. Analyzing
                  </Badge>
                </div>
              </div>
            )}




            



          </div>
        </CardContent>
      </Card>

      {/* Job Info */}
      <Card>
        <CardContent className="space-y-2 pt-6">
          <div>
            <p className="text-sm text-muted-foreground">Video Name</p>
            <p className="font-medium">{job.video_name}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Created</p>
            <p className="font-medium">
              {new Date(job.created_at).toLocaleString()}
            </p>
          </div>
          {job.started_at && (
            <div>
              <p className="text-sm text-muted-foreground">Started</p>
              <p className="font-medium">
                {new Date(job.started_at).toLocaleString()}
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
