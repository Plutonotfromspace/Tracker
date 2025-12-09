"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { mockJobs } from "@/lib/mock-data";
import { Job } from "@/lib/types";
import { ArrowUpDown, Search, Filter } from "lucide-react";
import Link from "next/link";
import { useState, useMemo, useEffect } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function getStatusBadge(status: Job["status"]) {
  const variants = {
    created: { variant: "secondary" as const, label: "Queued" },
    processing: { variant: "default" as const, label: "Processing" },
    completed: { variant: "default" as const, label: "Completed" },
    failed: { variant: "destructive" as const, label: "Failed" },
  };

  return (
    <Badge 
      variant={variants[status].variant}
      className={status === "processing" ? "animate-pulse bg-blue-600 hover:bg-blue-600" : ""}
    >
      {variants[status].label}
    </Badge>
  );
}

function formatDate(date: string | undefined) {
  if (!date) return "—";
  return new Date(date).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatDuration(start: string | undefined, end: string | undefined) {
  if (!start || !end) return "—";
  const duration = new Date(end).getTime() - new Date(start).getTime();
  const minutes = Math.floor(duration / 60000);
  const seconds = Math.floor((duration % 60000) / 1000);
  return `${minutes}m ${seconds}s`;
}

type SortField = "created_at" | "truck_count" | "video_name";
type SortOrder = "asc" | "desc";

export default function JobsPage() {
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState<Job["status"] | "all">("all");
  const [sortField, setSortField] = useState<SortField>("created_at");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [dataSource, setDataSource] = useState<"live" | "mock">("mock");

  useEffect(() => {
    const source = localStorage.getItem("data_source") as "live" | "mock" || "mock";
    setDataSource(source);

    const fetchJobs = async () => {
      if (source === "live") {
        try {
          const token = localStorage.getItem("auth_token");
          const response = await fetch(`${API_URL}/api/jobs/`, {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          });

          if (response.ok) {
            const data = await response.json();
            setJobs(data);
          } else {
            console.error("Failed to fetch jobs");
            setJobs([]);
          }
        } catch (error) {
          console.error("Error fetching jobs:", error);
          setJobs([]);
        } finally {
          setIsLoading(false);
        }
      } else {
        // Use mock data
        setJobs(mockJobs);
        setIsLoading(false);
      }
    };

    // Initial fetch
    fetchJobs();

    // Poll for updates every 3 seconds if there are active jobs
    const pollInterval = setInterval(() => {
      if (source === "live") {
        fetchJobs();
      }
    }, 3000);

    return () => clearInterval(pollInterval);
  }, []);

  const filteredAndSortedJobs = useMemo(() => {
    const filtered = jobs.filter((job) => {
      const matchesSearch =
        job.video_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        job.id.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStatus = statusFilter === "all" || job.status === statusFilter;
      return matchesSearch && matchesStatus;
    });

    filtered.sort((a, b) => {
      let comparison = 0;
      if (sortField === "created_at") {
        comparison = new Date(a.created_at || 0).getTime() - new Date(b.created_at || 0).getTime();
      } else if (sortField === "truck_count") {
        comparison = (a.truck_count || 0) - (b.truck_count || 0);
      } else if (sortField === "video_name") {
        comparison = a.video_name.localeCompare(b.video_name);
      }
      return sortOrder === "asc" ? comparison : -comparison;
    });

    return filtered;
  }, [searchTerm, statusFilter, sortField, sortOrder, jobs]);

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("desc");
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">All Jobs</h1>
          <p className="text-muted-foreground">
            View and filter all video processing jobs
          </p>
        </div>
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
            <p className="mt-4 text-muted-foreground">Loading jobs...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">All Jobs</h1>
        <p className="text-muted-foreground">
          View and filter all video processing jobs
        </p>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Filters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search by job ID or video name..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-9"
              />
            </div>
            <div className="flex flex-wrap gap-2">
              <Button
                variant={statusFilter === "all" ? "default" : "outline"}
                size="sm"
                className="flex-1 sm:flex-none"
                onClick={() => setStatusFilter("all")}
              >
                All
              </Button>
              <Button
                variant={statusFilter === "created" ? "default" : "outline"}
                size="sm"
                className="flex-1 sm:flex-none"
                onClick={() => setStatusFilter("created")}
              >
                Queued
              </Button>
              <Button
                variant={statusFilter === "processing" ? "default" : "outline"}
                size="sm"
                className="flex-1 sm:flex-none"
                onClick={() => setStatusFilter("processing")}
              >
                Processing
              </Button>
              <Button
                variant={statusFilter === "completed" ? "default" : "outline"}
                size="sm"
                className="flex-1 sm:flex-none"
                onClick={() => setStatusFilter("completed")}
              >
                Completed
              </Button>
              <Button
                variant={statusFilter === "failed" ? "default" : "outline"}
                size="sm"
                className="flex-1 sm:flex-none"
                onClick={() => setStatusFilter("failed")}
              >
                Failed
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      <Card>
        <CardHeader>
          <CardTitle>
            {filteredAndSortedJobs.length} {filteredAndSortedJobs.length === 1 ? "Job" : "Jobs"}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Job ID</TableHead>
                <TableHead>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="-ml-3 h-8"
                    onClick={() => toggleSort("video_name")}
                  >
                    Video Name
                    <ArrowUpDown className="ml-2 h-4 w-4" />
                  </Button>
                </TableHead>
                <TableHead>Status</TableHead>
                <TableHead>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="-ml-3 h-8"
                    onClick={() => toggleSort("truck_count")}
                  >
                    Trucks
                    <ArrowUpDown className="ml-2 h-4 w-4" />
                  </Button>
                </TableHead>
                <TableHead>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="-ml-3 h-8"
                    onClick={() => toggleSort("created_at")}
                  >
                    Created
                    <ArrowUpDown className="ml-2 h-4 w-4" />
                  </Button>
                </TableHead>
                <TableHead>Duration</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredAndSortedJobs.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center text-muted-foreground">
                    No jobs found matching your filters
                  </TableCell>
                </TableRow>
              ) : (
                filteredAndSortedJobs.map((job) => (
                  <TableRow 
                    key={job.id}
                    className={job.status !== "failed" ? "cursor-pointer hover:bg-muted/50" : "cursor-not-allowed opacity-60"}
                    onClick={() => {
                      if (job.status !== "failed") {
                        window.location.href = `/jobs/${job.id}`;
                      }
                    }}
                  >
                    <TableCell className="font-mono text-sm">
                      <span className={job.status !== "failed" ? "text-blue-600 hover:underline" : "text-muted-foreground"}>
                        {job.id.slice(0, 13)}...
                      </span>
                    </TableCell>
                    <TableCell className="max-w-xs truncate">
                      {job.video_name}
                    </TableCell>
                    <TableCell>{getStatusBadge(job.status)}</TableCell>
                    <TableCell>
                      {job.truck_count ? (
                        <span className="font-medium">{job.truck_count}</span>
                      ) : (
                        "—"
                      )}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatDate(job.created_at)}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatDuration(job.started_at, job.completed_at)}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
