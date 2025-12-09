"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useState, useCallback, useEffect } from "react";
import { Upload, FileVideo, X, CheckCircle2, Lock } from "lucide-react";
import { useRouter } from "next/navigation";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string>("");
  const [showDemoModal, setShowDemoModal] = useState(false);
  const [isDemoAccount, setIsDemoAccount] = useState(false);
  const [isMockMode, setIsMockMode] = useState(false);
  const [videoSelected, setVideoSelected] = useState(false);

  useEffect(() => {
    // Check if user is in mock data mode
    const dataSource = localStorage.getItem("data_source");
    const isMock = dataSource === "mock";
    setIsMockMode(isMock);
    
    // Also check if demo account for backwards compatibility
    const isDemo = localStorage.getItem("is_demo") === "true";
    setIsDemoAccount(isDemo);
  }, []);

  const validateFile = (file: File): boolean => {
    const validTypes = ["video/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo"];
    const maxSize = 5 * 1024 * 1024 * 1024; // 5GB

    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|mpeg|mov|avi)$/i)) {
      setError("Please upload a valid video file (MP4, MPEG, MOV, or AVI)");
      return false;
    }

    if (file.size > maxSize) {
      setError("File size must be less than 5GB");
      return false;
    }

    setError("");
    return true;
  };

  const handleFileSelect = useCallback((selectedFile: File) => {
    // Check if in mock mode - cannot upload in mock mode
    if (isMockMode) {
      setShowDemoModal(true);
      return;
    }
    
    // Check if demo account trying to upload
    if (isDemoAccount) {
      setShowDemoModal(true);
      return;
    }
    
    if (validateFile(selectedFile)) {
      setFile(selectedFile);
    }
  }, [isMockMode, isDemoAccount]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (isMockMode || isDemoAccount) {
      setShowDemoModal(true);
      return;
    }
    setIsDragging(true);
  }, [isMockMode, isDemoAccount]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    if (isMockMode || isDemoAccount) {
      setShowDemoModal(true);
      return;
    }

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFileSelect(droppedFile);
    }
  }, [handleFileSelect, isMockMode, isDemoAccount]);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      handleFileSelect(selectedFile);
    }
  };

  const handleBrowseClick = (e: React.MouseEvent) => {
    if (isMockMode || isDemoAccount) {
      e.preventDefault();
      setShowDemoModal(true);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setUploadProgress(0);

    // Simulate upload progress
    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(() => {
            // Mock: redirect to a new job
            router.push("/jobs");
          }, 500);
          return 100;
        }
        return prev + 10;
      });
    }, 300);

    // TODO: Replace with actual API call to backend
    // const formData = new FormData();
    // formData.append('video', file);
    // const response = await fetch('/api/jobs', {
    //   method: 'POST',
    //   body: formData
    // });
  };

  const handleRemove = () => {
    setFile(null);
    setError("");
    setUploadProgress(0);
  };

  const handleVideoSelect = () => {
    setVideoSelected(true);
  };

  const handleStartProcessing = async () => {
    setShowDemoModal(false);
    setIsUploading(true);
    setUploadProgress(0);
    
    try {
      // Call the demo upload endpoint with 3000 max frames limit
      const url = new URL("http://localhost:8000/api/upload/demo");
      url.searchParams.append("max_frames", "3000");
      
      const response = await fetch(url.toString(), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to create demo job");
      }

      const data = await response.json();
      
      // Simulate progress for better UX
      setUploadProgress(100);
      
      // Redirect to the job detail page
      setTimeout(() => {
        router.push(`/jobs/${data.job_id}`);
      }, 500);
      
    } catch (err) {
      console.error("Demo upload error:", err);
      setError(err instanceof Error ? err.message : "Failed to process demo video");
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + " " + sizes[i];
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Upload Video</h1>
        <p className="text-muted-foreground">
          Upload a video file to detect and track trucks
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Select Video File</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Drop Zone */}
          {!file && (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={(e) => {
                if (isMockMode || isDemoAccount) {
                  e.preventDefault();
                  setShowDemoModal(true);
                  return;
                }
                document.getElementById('file-upload')?.click();
              }}
              className={`
                border-2 border-dashed rounded-lg p-12 text-center transition-colors cursor-pointer
                ${isDragging 
                  ? "border-primary bg-primary/5" 
                  : "border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/50"
                }
              `}
            >
              <div className="flex flex-col items-center gap-4">
                <div className="rounded-full bg-primary/10 p-4">
                  <Upload className="h-8 w-8 text-primary" />
                </div>
                <div>
                  <p className="text-lg font-medium">
                    Drop your video file here
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">
                    or click to browse
                  </p>
                </div>
                <Input
                  type="file"
                  accept="video/*"
                  onChange={handleFileInput}
                  className="hidden"
                  id="file-upload"
                />
                <Button 
                  variant="outline" 
                  type="button"
                >
                  Browse Files
                </Button>
                <p className="text-xs text-muted-foreground">
                  Supports MP4, MPEG, MOV, AVI up to 5GB
                </p>
              </div>
            </div>
          )}

          {/* File Preview */}
          {file && (
            <div className="space-y-4">
              <div className="flex items-start gap-4 rounded-lg border p-4">
                <div className="rounded-lg bg-primary/10 p-3">
                  <FileVideo className="h-6 w-6 text-primary" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{file.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {formatFileSize(file.size)}
                  </p>
                </div>
                {!isUploading && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleRemove}
                    className="shrink-0"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                )}
              </div>

              {/* Upload Progress */}
              {isUploading && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Uploading...</span>
                    <span className="font-medium">{uploadProgress}%</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
                    <div
                      className="h-full bg-primary transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                </div>
              )}

              {uploadProgress === 100 && (
                <div className="flex items-center gap-2 text-sm text-green-600">
                  <CheckCircle2 className="h-4 w-4" />
                  <span>Upload complete! Redirecting...</span>
                </div>
              )}
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="rounded-lg bg-destructive/10 p-4 text-sm text-destructive">
              {error}
            </div>
          )}

          {/* Actions */}
          {file && !isUploading && uploadProgress === 0 && (
            <div className="flex gap-2">
              <Button onClick={handleUpload} className="flex-1">
                <Upload className="mr-2 h-4 w-4" />
                Start Processing
              </Button>
              <Button variant="outline" onClick={handleRemove}>
                Cancel
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Info Card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">How it works</CardTitle>
        </CardHeader>
        <CardContent>
          <ol className="space-y-3 text-sm text-muted-foreground">
            <li className="flex gap-3">
              <span className="font-semibold text-foreground">1.</span>
              <div>
                <span className="text-foreground font-medium">Upload a video file from the approved camera only</span>
                <p className="mt-1 text-xs text-orange-600 dark:text-orange-500 font-medium">
                  Videos must be from the designated World Trade Bridge camera.
                </p>
              </div>
            </li>
            <li className="flex gap-3">
              <span className="font-semibold text-foreground">2.</span>
              <span>Our AI will process the video and detect all trucks</span>
            </li>
            <li className="flex gap-3">
              <span className="font-semibold text-foreground">3.</span>
              <span>View detailed results including truck counts, body types, and timestamps</span>
            </li>
            <li className="flex gap-3">
              <span className="font-semibold text-foreground">4.</span>
              <span>Download analysis reports for your records</span>
            </li>
          </ol>
        </CardContent>
      </Card>

      {/* Demo Account Restriction Modal */}
      <Dialog open={showDemoModal} onOpenChange={setShowDemoModal}>
        <DialogContent className="sm:max-w-md transition-all duration-300 ease-in-out">
          <DialogHeader>
            <DialogTitle>{isMockMode ? "Mock Data Mode" : "Select a Video File"}</DialogTitle>
            <DialogDescription>
              {isMockMode ? "Video upload not available" : "Choose a video to process"}
            </DialogDescription>
          </DialogHeader>

          {/* Mock mode restriction - cannot upload */}
          {isMockMode ? (
            <div className="flex items-start gap-3 p-4 bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-900 rounded-lg">
              <Lock className="h-5 w-5 text-amber-600 dark:text-amber-500 shrink-0 mt-0.5" />
              <div className="flex-1 text-sm">
                <p className="font-medium text-amber-900 dark:text-amber-200">Upload Not Available in Mock Mode</p>
                <p className="text-amber-800 dark:text-amber-300 mt-2">
                  You are currently viewing the application in mock data mode. Video uploads are not available in this mode.
                </p>
                <p className="text-amber-800 dark:text-amber-300 mt-2">
                  To upload and process videos, please switch to live mode from the settings.
                </p>
              </div>
            </div>
          ) : (
            <>
              {/* Demo restriction notice */}
              <div className="flex items-start gap-3 p-3 bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-900 rounded-lg">
                <Lock className="h-5 w-5 text-amber-600 dark:text-amber-500 shrink-0 mt-0.5" />
                <div className="flex-1 text-sm">
                  <p className="font-medium text-amber-900 dark:text-amber-200">Demo Account Restriction</p>
                  <p className="text-amber-800 dark:text-amber-300 mt-1">
                    During the demo phase, only the sample video is available for processing.
                  </p>
                </div>
              </div>

            <div className="border rounded-lg">
              {/* File explorer header */}
              <div className="bg-muted/50 border-b px-4 py-2 text-sm font-medium">
                Available Videos
              </div>
            
            {/* File list */}
            <div className="p-2">
              <button
                onClick={handleVideoSelect}
                className="w-full flex items-center gap-3 px-3 py-2 rounded hover:bg-muted/50 transition-colors text-left group"
              >
                <FileVideo className="h-5 w-5 text-primary shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm group-hover:text-primary transition-colors">
                    World Trade Bridge Oct 16th from 3.22PM to 3.51PM.mp4
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Video file • Ready to process
                  </div>
                </div>
              </button>
            </div>
          </div>


            </>
          )}

          <div className="flex justify-end gap-2 mt-4">
            <Button onClick={() => {
              setShowDemoModal(false);
              setVideoSelected(false);
            }} variant="outline">
              {isMockMode ? "Close" : "Cancel"}
            </Button>
            {!isMockMode && videoSelected && (
              <Button onClick={handleStartProcessing}>
                Start Processing
              </Button>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
