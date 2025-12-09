"use client";

import { use, useState, useMemo, useEffect } from "react";
import { notFound } from "next/navigation";
import { mockJobDetails } from "@/lib/mock-data";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { JobProgressView } from "@/components/job-progress-view";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
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
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Bar, BarChart, CartesianGrid } from "recharts";
import {
  CheckCircle2,
  Clock,
  XCircle,
  FileVideo,
  TruckIcon,
  Download,
  ArrowLeft,
  Filter,
  X as XIcon,
  ChevronDown,
  FileDown,
  Trash2,
} from "lucide-react";
import Link from "next/link";
import { Job, Truck } from "@/lib/types";

function getStatusBadge(status: Job["status"]) {
  const variants = {
    created: { variant: "secondary" as const, label: "Queued", icon: Clock },
    processing: { variant: "outline" as const, label: "Processing", icon: Clock },
    completed: { variant: "default" as const, label: "Completed", icon: CheckCircle2 },
    failed: { variant: "destructive" as const, label: "Failed", icon: XCircle },
  };

  const config = variants[status];
  const Icon = config.icon;

  return (
    <Badge variant={config.variant}>
      <Icon className="mr-1 h-3 w-3" />
      {config.label}
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

function getBodyTypeLabel(bodyType: string): string {
  const labels: Record<string, string> = {
    dry_van: "Dry Van",
    reefer: "Reefer",
    flatbed: "Flatbed",
  };
  return labels[bodyType] || bodyType;
}

function getSmallVehicleLabel(vehicleType: string): string {
  const labels: Record<string, string> = {
    bobtail: "Bobtail",
    box_truck: "Box Truck",
    pickup: "Pickup",
    van: "Van",
    other: "Other",
  };
  return labels[vehicleType] || vehicleType;
}

function getVehicleTypeColor(type: string): string {
  const colors: Record<string, string> = {
    // Truck body types
    "Dry Van": "hsl(210, 100%, 56%)",      // Blue
    "Reefer": "hsl(142, 71%, 45%)",        // Green
    "Flatbed": "hsl(24, 95%, 53%)",        // Orange
    // Small vehicle types
    "Bobtail": "hsl(271, 91%, 65%)",       // Purple
    "Box Truck": "hsl(49, 98%, 60%)",      // Yellow
    "Pickup": "hsl(340, 82%, 52%)",        // Pink/Red
    "Van": "hsl(180, 77%, 47%)",           // Cyan
    "Other": "hsl(0, 0%, 63%)",            // Gray
  };
  return colors[type] || "hsl(var(--chart-1))";
}

export default function JobDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const [jobDetails, setJobDetails] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedCropImage, setSelectedCropImage] = useState<string | null>(null);
  const [selectedTruckId, setSelectedTruckId] = useState<string | null>(null);

  // Filter states
  const [selectedBodyTypes, setSelectedBodyTypes] = useState<string[]>([]);
  const [selectedAxleTypes, setSelectedAxleTypes] = useState<string[]>([]);
  const [selectedSmallVehicleTypes, setSelectedSmallVehicleTypes] = useState<string[]>([]);
  const [confidenceRange, setConfidenceRange] = useState<number[]>([0, 100]);
  const [searchTerm, setSearchTerm] = useState("");
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const [colorMode, setColorMode] = useState(false);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportFormat, setExportFormat] = useState<'csv' | 'json'>('csv');
  const [exportFilters, setExportFilters] = useState({
    bodyTypes: [] as string[],
    axleTypes: [] as string[],
    smallVehicleTypes: [] as string[],
    confidenceRange: [0, 100] as number[],
  });
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  // Filter trucks based on all active filters - must be before any conditional returns
  const filteredTrucks = useMemo(() => {
    if (!jobDetails || !jobDetails.trucks) return [];

    return jobDetails.trucks.filter((truck: any) => {
      // Vehicle type filter (body type OR small vehicle type)
      // If either filter is active, vehicle must match at least one selected type
      const hasVehicleTypeFilter = selectedBodyTypes.length > 0 || selectedSmallVehicleTypes.length > 0;
      
      if (hasVehicleTypeFilter) {
        const matchesBodyType = selectedBodyTypes.length > 0 && 
          truck.body_type && selectedBodyTypes.includes(truck.body_type);
        const matchesSmallVehicle = selectedSmallVehicleTypes.length > 0 && 
          truck.small_vehicle_type && selectedSmallVehicleTypes.includes(truck.small_vehicle_type);
        
        if (!matchesBodyType && !matchesSmallVehicle) {
          return false;
        }
      }

      // Axle type filter
      if (selectedAxleTypes.length > 0) {
        if (!truck.axle_type || !selectedAxleTypes.includes(truck.axle_type)) {
          return false;
        }
      }

      // Confidence filter
      const confidence = truck.confidence * 100;
      if (confidence < confidenceRange[0] || confidence > confidenceRange[1]) {
        return false;
      }

      // Search filter
      if (searchTerm) {
        const search = searchTerm.toLowerCase();
        const matchesTruckId = truck.truck_id?.toLowerCase().includes(search);
        const matchesBodyType = truck.body_type?.toLowerCase().includes(search);
        const matchesAxleType = truck.axle_type?.toLowerCase().includes(search);
        const matchesSmallVehicle = truck.small_vehicle_type?.toLowerCase().includes(search);
        
        if (!matchesTruckId && !matchesBodyType && !matchesAxleType && !matchesSmallVehicle) {
          return false;
        }
      }

      return true;
    });
  }, [jobDetails, selectedBodyTypes, selectedAxleTypes, selectedSmallVehicleTypes, confidenceRange, searchTerm]);

  // Get filtered vehicle distribution for chart
  const filteredVehicleDistribution = useMemo(() => {
    const distribution: Record<string, number> = {};
    
    filteredTrucks.forEach((truck: any) => {
      let label: string;
      
      if (truck.body_type) {
        label = getBodyTypeLabel(truck.body_type);
      } else if (truck.small_vehicle_type) {
        label = getSmallVehicleLabel(truck.small_vehicle_type);
      } else {
        label = "Unknown";
      }
      
      distribution[label] = (distribution[label] || 0) + 1;
    });

    return Object.entries(distribution).map(([type, count]) => ({
      type: type,
      count,
      fill: getVehicleTypeColor(type),
    }));
  }, [filteredTrucks]);

  useEffect(() => {
    const dataSource = localStorage.getItem("data_source") as "live" | "mock" || "mock";

    if (dataSource === "live") {
      // Fetch from API
      const fetchJobDetails = async () => {
        try {
          const token = localStorage.getItem("auth_token");
          
          // Fetch job info
          const jobResponse = await fetch(`${API_URL}/api/jobs/${id}`, {
            headers: { Authorization: `Bearer ${token}` },
          });

          if (!jobResponse.ok) {
            notFound();
            return;
          }

          const job = await jobResponse.json();

          // Fetch trucks for this job
          const trucksResponse = await fetch(`${API_URL}/api/trucks/job/${id}`, {
            headers: { Authorization: `Bearer ${token}` },
          });

          const trucks = trucksResponse.ok ? await trucksResponse.json() : [];

          setJobDetails({
            ...job,
            trucks: trucks,
          });
        } catch (error) {
          console.error("Error fetching job details:", error);
          notFound();
        } finally {
          setIsLoading(false);
        }
      };
      fetchJobDetails();
    } else {
      // Use mock data
      const mockData = mockJobDetails[id];
      if (!mockData) {
        notFound();
        return;
      }
      setJobDetails(mockData);
      setIsLoading(false);
    }
  }, [id]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Loading job details...</p>
        </div>
      </div>
    );
  }

  if (!jobDetails) {
    notFound();
  }

  // Show progress view for queued or processing jobs
  if (jobDetails.status === "created" || jobDetails.status === "processing") {
    return <JobProgressView initialJob={jobDetails} />;
  }

  // Show error message for failed jobs
  if (jobDetails.status === "failed") {
    return (
      <div className="space-y-6">
        <div>
          <Button variant="ghost" size="sm" asChild className="mb-4">
            <Link href="/jobs">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Jobs
            </Link>
          </Button>
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">Job Failed</h1>
              <p className="text-muted-foreground mt-1 font-mono text-sm">
                {jobDetails.id}
              </p>
            </div>
            <Badge variant="destructive">Failed</Badge>
          </div>
        </div>
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle className="text-destructive">Error</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm">{jobDetails.error || "An error occurred during processing."}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Job Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div>
              <p className="text-sm text-muted-foreground">Video Name</p>
              <p className="font-medium">{jobDetails.video_name}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Created</p>
              <p className="font-medium">
                {new Date(jobDetails.created_at).toLocaleString()}
              </p>
            </div>
            {jobDetails.started_at && (
              <div>
                <p className="text-sm text-muted-foreground">Started</p>
                <p className="font-medium">
                  {new Date(jobDetails.started_at).toLocaleString()}
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    );
  }

  // Prepare chart data from filtered results, sorted by count (descending)
  const chartData = filteredVehicleDistribution
    .sort((a, b) => b.count - a.count)
    .map((item) => ({
      bodyType: item.type,
      count: item.count,
      fill: colorMode ? item.fill : "var(--color-count)",
    }));

  const chartConfig = {
    count: {
      label: "Count",
      color: "hsl(var(--chart-1))",
    },
  } satisfies ChartConfig;

  const bodyTypeOptions = ["dry_van", "reefer", "flatbed"];
  const axleTypeOptions = ["standard", "spread"];
  const smallVehicleOptions = ["bobtail", "box_truck", "pickup", "van", "other"];

  const activeFiltersCount =
    selectedBodyTypes.length +
    selectedAxleTypes.length +
    selectedSmallVehicleTypes.length +
    (confidenceRange[0] !== 0 || confidenceRange[1] !== 100 ? 1 : 0) +
    (searchTerm ? 1 : 0);

  const clearAllFilters = () => {
    setSelectedBodyTypes([]);
    setSelectedAxleTypes([]);
    setSelectedSmallVehicleTypes([]);
    setConfidenceRange([0, 100]);
    setSearchTerm("");
  };

  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      const dataSource = localStorage.getItem("data_source") as "live" | "mock" || "mock";
      
      if (dataSource === "live") {
        const token = localStorage.getItem("auth_token");
        const response = await fetch(`${API_URL}/api/jobs/${id}`, {
          method: 'DELETE',
          headers: { 
            Authorization: `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          throw new Error('Failed to delete job');
        }
      }
      
      // Redirect to jobs list after successful deletion
      window.location.href = '/jobs';
    } catch (error) {
      console.error('Error deleting job:', error);
      alert('Failed to delete job. Please try again.');
      setIsDeleting(false);
      setDeleteDialogOpen(false);
    }
  };

  const handleExport = () => {
    // Filter trucks based on export filter settings
    const trucksToExport = jobDetails.trucks.filter((truck: any) => {
      // Body type filter
      if (exportFilters.bodyTypes.length > 0) {
        const matchesBodyType = truck.body_type && exportFilters.bodyTypes.includes(truck.body_type);
        const matchesSmallVehicle = truck.small_vehicle_type && exportFilters.smallVehicleTypes.includes(truck.small_vehicle_type);
        if (!matchesBodyType && !matchesSmallVehicle) return false;
      }

      // Small vehicle filter (if not already matched by body type)
      if (exportFilters.smallVehicleTypes.length > 0 && exportFilters.bodyTypes.length === 0) {
        if (!truck.small_vehicle_type || !exportFilters.smallVehicleTypes.includes(truck.small_vehicle_type)) {
          return false;
        }
      }

      // Axle type filter
      if (exportFilters.axleTypes.length > 0) {
        if (!truck.axle_type || !exportFilters.axleTypes.includes(truck.axle_type)) {
          return false;
        }
      }

      // Confidence filter
      const confidence = truck.confidence * 100;
      if (confidence < exportFilters.confidenceRange[0] || confidence > exportFilters.confidenceRange[1]) {
        return false;
      }

      return true;
    });

    if (exportFormat === 'csv') {
      // Generate CSV
      const headers = ['Vehicle ID', 'Truck ID', 'Timestamp', 'Body Type', 'Axle Type', 'Small Vehicle Type', 'Confidence'];
      const rows = trucksToExport.map((truck: any) => [
        truck.unique_truck_id || '',
        truck.truck_id || '',
        truck.timestamp || '',
        truck.body_type || '',
        truck.axle_type || '',
        truck.small_vehicle_type || '',
        truck.confidence ? (truck.confidence * 100).toFixed(2) + '%' : '',
      ]);
      
      const csvContent = [
        headers.join(','),
        ...rows.map((row: string[]) => row.map((cell: string) => `"${cell}"`).join(','))
      ].join('\n');
      
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${jobDetails.id}_trucks.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } else {
      // Generate JSON
      const jsonContent = JSON.stringify(trucksToExport, null, 2);
      const blob = new Blob([jsonContent], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${jobDetails.id}_trucks.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    }
    
    setExportDialogOpen(false);
  };

  return (
    <div className="space-y-6 px-4 sm:px-6">
      {/* Header */}
      <div>
        <Button variant="ghost" size="sm" asChild className="mb-4">
          <Link href="/jobs">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Jobs
          </Link>
        </Button>
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Job Details</h1>
            <p className="text-muted-foreground mt-1 font-mono text-sm">
              {jobDetails.id}
            </p>
          </div>
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
            <div className="flex flex-wrap gap-2">
              <Button 
                variant="outline" 
                size="sm"
                className="flex-1 sm:flex-none"
                onClick={() => {
                  setExportFormat('csv');
                  setExportFilters({
                    bodyTypes: selectedBodyTypes,
                    axleTypes: selectedAxleTypes,
                    smallVehicleTypes: selectedSmallVehicleTypes,
                    confidenceRange: confidenceRange,
                  });
                  setExportDialogOpen(true);
                }}
                disabled={!jobDetails.trucks || jobDetails.trucks.length === 0}
              >
                <FileDown className="mr-2 h-4 w-4" />
                Export CSV
              </Button>
              <Button 
                variant="outline" 
                size="sm"
                className="flex-1 sm:flex-none"
                onClick={() => {
                  setExportFormat('json');
                  setExportFilters({
                    bodyTypes: selectedBodyTypes,
                    axleTypes: selectedAxleTypes,
                    smallVehicleTypes: selectedSmallVehicleTypes,
                    confidenceRange: confidenceRange,
                  });
                  setExportDialogOpen(true);
                }}
                disabled={!jobDetails.trucks || jobDetails.trucks.length === 0}
              >
                <FileDown className="mr-2 h-4 w-4" />
                Export JSON
              </Button>
              <Button 
                variant="destructive" 
                size="sm"
                className="flex-1 sm:flex-none"
                onClick={() => setDeleteDialogOpen(true)}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Delete Job
              </Button>
            </div>
            {getStatusBadge(jobDetails.status)}
          </div>
        </div>
      </div>

      {/* Job Info Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Video Name</CardTitle>
            <FileVideo className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent className="p-4 sm:p-6">
            <div className="text-sm font-medium truncate">
              {jobDetails.video_name}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Trucks Detected</CardTitle>
            <TruckIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent className="p-4 sm:p-6">
            <div className="text-2xl font-bold">{jobDetails.truck_count || 0}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Processing Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent className="p-4 sm:p-6">
            <div className="text-2xl font-bold">
              {formatDuration(jobDetails.started_at, jobDetails.completed_at)}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Created</CardTitle>
          </CardHeader>
          <CardContent className="p-4 sm:p-6">
            <div className="text-sm font-medium">
              {formatDate(jobDetails.created_at)}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Vehicle Type Distribution Chart */}
      {filteredTrucks.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Vehicle Type Distribution</CardTitle>
                <p className="text-sm text-muted-foreground">
                  {activeFiltersCount > 0 ? "Showing filtered results" : "All detected vehicles"}
                </p>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="color-mode"
                  checked={colorMode}
                  onCheckedChange={setColorMode}
                />
                <Label htmlFor="color-mode" className="text-sm font-normal cursor-pointer">
                  Color by Type
                </Label>
              </div>
            </div>
          </CardHeader>
          <CardContent className="px-2 sm:p-6">
            <ChartContainer config={chartConfig} className="aspect-auto h-[250px] w-full sm:h-[300px]">
              <BarChart 
                accessibilityLayer 
                data={chartData}
                margin={{ left: 0, right: 0, top: 10, bottom: 0 }}
              >
                <CartesianGrid vertical={false} />
                <ChartTooltip 
                  content={<ChartTooltipContent 
                    labelFormatter={(value, payload) => {
                      return payload?.[0]?.payload?.bodyType || value;
                    }}
                  />} 
                />
                <Bar
                  dataKey="count"
                  radius={8}
                />
              </BarChart>
            </ChartContainer>
          </CardContent>
        </Card>
      )}

      {/* Advanced Filters */}
      <Card>
        <Collapsible open={isFilterOpen} onOpenChange={setIsFilterOpen}>
          <CardHeader>
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 flex-wrap">
                <Filter className="h-4 w-4" />
                <CardTitle>Advanced Filters</CardTitle>
                {activeFiltersCount > 0 && (
                  <Badge variant="secondary">{activeFiltersCount} active</Badge>
                )}
              </div>
              <div className="flex items-center gap-2 shrink-0">
                {activeFiltersCount > 0 && (
                  <Button variant="ghost" size="sm" onClick={clearAllFilters} className="hidden sm:flex">
                    <XIcon className="mr-2 h-4 w-4" />
                    Clear All
                  </Button>
                )}
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" size="sm">
                    <ChevronDown className="h-4 w-4" />
                  </Button>
                </CollapsibleTrigger>
              </div>
            </div>
          </CardHeader>
          <CollapsibleContent>
            <CardContent className="space-y-6">
              <div className="grid gap-6 md:grid-cols-2">
                {/* Body Type Filter */}
                <div className="space-y-3">
                  <Label className="text-sm font-medium">Body Type</Label>
                  <div className="space-y-2">
                    {bodyTypeOptions.map((type) => (
                      <div key={type} className="flex items-center space-x-2">
                        <Checkbox
                          id={`body-${type}`}
                          checked={selectedBodyTypes.includes(type)}
                          onCheckedChange={(checked) => {
                            if (checked) {
                              setSelectedBodyTypes([...selectedBodyTypes, type]);
                            } else {
                              setSelectedBodyTypes(
                                selectedBodyTypes.filter((t) => t !== type)
                              );
                            }
                          }}
                        />
                        <label
                          htmlFor={`body-${type}`}
                          className="text-sm cursor-pointer"
                        >
                          {getBodyTypeLabel(type)}
                        </label>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Axle Type Filter */}
                <div className="space-y-3">
                  <Label className="text-sm font-medium">Axle Type</Label>
                  <div className="space-y-2">
                    {axleTypeOptions.map((type) => (
                      <div key={type} className="flex items-center space-x-2">
                        <Checkbox
                          id={`axle-${type}`}
                          checked={selectedAxleTypes.includes(type)}
                          onCheckedChange={(checked) => {
                            if (checked) {
                              setSelectedAxleTypes([...selectedAxleTypes, type]);
                            } else {
                              setSelectedAxleTypes(
                                selectedAxleTypes.filter((t) => t !== type)
                              );
                            }
                          }}
                        />
                        <label
                          htmlFor={`axle-${type}`}
                          className="text-sm cursor-pointer capitalize"
                        >
                          {type}
                        </label>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Small Vehicle Type Filter */}
              <div className="col-span-2 grid gap-4 grid-cols-2">
                <div className="space-y-3">
                  <Label className="text-sm font-medium">Small Vehicle Type</Label>
                  <div className="space-y-2">
                    {smallVehicleOptions.map((type) => (
                      <div key={type} className="flex items-center space-x-2">
                        <Checkbox
                          id={`small-${type}`}
                          checked={selectedSmallVehicleTypes.includes(type)}
                          onCheckedChange={(checked) => {
                            if (checked) {
                              setSelectedSmallVehicleTypes([...selectedSmallVehicleTypes, type]);
                            } else {
                              setSelectedSmallVehicleTypes(
                                selectedSmallVehicleTypes.filter((t) => t !== type)
                              );
                            }
                          }}
                        />
                        <label
                          htmlFor={`small-${type}`}
                          className="text-sm cursor-pointer"
                        >
                          {getSmallVehicleLabel(type)}
                        </label>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Confidence Range */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="text-sm font-medium">
                    Confidence Range
                  </Label>
                  <span className="text-sm text-muted-foreground">
                    {confidenceRange[0]}% - {confidenceRange[1]}%
                  </span>
                </div>
                <Slider
                  min={0}
                  max={100}
                  step={5}
                  value={confidenceRange}
                  onValueChange={setConfidenceRange}
                  className="w-full"
                />
              </div>

              {/* Search */}
              <div className="space-y-3">
                <Label htmlFor="search" className="text-sm font-medium">
                  Search Truck ID
                </Label>
                <Input
                  id="search"
                  placeholder="Search by truck ID..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>

              {/* Filtered Results Count */}
              <div className="rounded-lg bg-muted p-3">
                <p className="text-sm text-muted-foreground">
                  Showing <span className="font-semibold text-foreground">{filteredTrucks.length}</span> of{" "}
                  <span className="font-semibold text-foreground">{jobDetails.trucks?.length || 0}</span> trucks
                </p>
              </div>
            </CardContent>
          </CollapsibleContent>
        </Collapsible>
      </Card>

      {/* Trucks Table */}
      <Card>
        <CardHeader>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <CardTitle>Detected Trucks</CardTitle>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                className="flex-1 sm:flex-none"
                onClick={() => {
                  setExportFormat('csv');
                  setExportFilters({
                    bodyTypes: selectedBodyTypes,
                    axleTypes: selectedAxleTypes,
                    smallVehicleTypes: selectedSmallVehicleTypes,
                    confidenceRange: confidenceRange,
                  });
                  setExportDialogOpen(true);
                }}
              >
                <Download className="mr-2 h-4 w-4" />
                Export CSV
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="flex-1 sm:flex-none"
                onClick={() => {
                  setExportFormat('json');
                  setExportFilters({
                    bodyTypes: selectedBodyTypes,
                    axleTypes: selectedAxleTypes,
                    smallVehicleTypes: selectedSmallVehicleTypes,
                    confidenceRange: confidenceRange,
                  });
                  setExportDialogOpen(true);
                }}
              >
                <Download className="mr-2 h-4 w-4" />
                Export JSON
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Vehicle ID</TableHead>
                <TableHead>Timestamp</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Axle Config</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredTrucks.length > 0 ? (
                filteredTrucks.map((truck) => (
                  <TableRow key={truck.unique_truck_id || truck.truck_id}>
                    <TableCell className="font-medium">
                      <div className="flex flex-col">
                        <span>{truck.unique_truck_id || truck.truck_id}</span>
                        {truck.unique_truck_id && (
                          <span className="text-xs text-muted-foreground">{truck.truck_id}</span>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatDate(truck.timestamp)}
                    </TableCell>
                    <TableCell>
                      <Badge 
                        variant="outline"
                        style={colorMode ? {
                          backgroundColor: `${truck.body_type
                            ? getVehicleTypeColor(getBodyTypeLabel(truck.body_type))
                            : truck.small_vehicle_type
                            ? getVehicleTypeColor(getSmallVehicleLabel(truck.small_vehicle_type))
                            : "hsl(var(--muted))"}15`,
                          borderColor: truck.body_type
                            ? getVehicleTypeColor(getBodyTypeLabel(truck.body_type))
                            : truck.small_vehicle_type
                            ? getVehicleTypeColor(getSmallVehicleLabel(truck.small_vehicle_type))
                            : "hsl(var(--border))",
                          color: truck.body_type
                            ? getVehicleTypeColor(getBodyTypeLabel(truck.body_type))
                            : truck.small_vehicle_type
                            ? getVehicleTypeColor(getSmallVehicleLabel(truck.small_vehicle_type))
                            : "hsl(var(--foreground))",
                        } : {}}
                      >
                        {truck.body_type
                          ? getBodyTypeLabel(truck.body_type)
                          : truck.small_vehicle_type
                          ? getSmallVehicleLabel(truck.small_vehicle_type)
                          : "Unknown"}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm capitalize">
                      {truck.axle_type || "—"}
                    </TableCell>
                    <TableCell className="text-sm">
                      {truck.confidence ? (truck.confidence * 100).toFixed(1) : "0"}%
                    </TableCell>
                    <TableCell>
                      <Button 
                        variant="ghost" 
                        size="sm"
                        onClick={() => {
                          // For live data: construct URL from crop_path
                          // For mock data: use files.crop which is already a path
                          const cropPath = truck.crop_path || truck.files?.crop;
                          if (cropPath) {
                            const imageUrl = truck.crop_path 
                              ? `${API_URL}/${cropPath.replace(/\\/g, '/')}`
                              : cropPath;
                            setSelectedCropImage(imageUrl);
                            setSelectedTruckId(truck.unique_truck_id || truck.truck_id);
                          }
                        }}
                        disabled={!truck.crop_path && !truck.files?.crop}
                      >
                        View Crop
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell
                    colSpan={6}
                    className="text-center text-muted-foreground"
                  >
                    {jobDetails.trucks && jobDetails.trucks.length > 0
                      ? "No vehicles match the current filters"
                      : "No vehicles detected"}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Error Message */}
      {jobDetails.error && (
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle className="text-destructive">Error</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-destructive">{jobDetails.error}</p>
          </CardContent>
        </Card>
      )}

      {/* Crop Image Modal */}
      <Dialog open={!!selectedCropImage} onOpenChange={(open: boolean) => !open && (setSelectedCropImage(null), setSelectedTruckId(null))}>
        <DialogContent className="max-w-7xl p-0 overflow-hidden gap-0">
          <DialogHeader className="px-6 pt-6 pb-4 border-b">
            <DialogTitle className="text-lg font-mono font-medium tracking-tight">{selectedTruckId || 'Vehicle Crop Image'}</DialogTitle>
          </DialogHeader>
          <div className="relative bg-muted/30 flex items-center justify-center min-h-[500px] p-6">
            {(() => {
              const dataSource = typeof window !== 'undefined' 
                ? (localStorage.getItem("data_source") as "live" | "mock" || "mock")
                : "mock";
              
              if (dataSource === "mock") {
                return (
                  <div className="flex flex-col items-center justify-center text-center space-y-4 p-8">
                    <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center">
                      <TruckIcon className="w-8 h-8 text-muted-foreground" />
                    </div>
                    <div>
                      <h3 className="text-lg font-medium mb-2">Mock Data Mode</h3>
                      <p className="text-sm text-muted-foreground max-w-md">
                        Vehicle crop images are not available in mock data mode. 
                        Switch to live mode to view actual vehicle images.
                      </p>
                    </div>
                  </div>
                );
              }
              
              return selectedCropImage && (
                <img 
                  src={selectedCropImage} 
                  alt="Vehicle crop" 
                  className="max-w-full max-h-[75vh] object-contain rounded-md shadow-xl border border-border/50"
                />
              );
            })()}
          </div>
          <div className="px-6 py-4 border-t bg-muted/20 flex justify-between items-center">
            <p className="text-sm text-muted-foreground">
              Click outside or press ESC to close
            </p>
            <Button 
              variant="outline" 
              size="sm"
              disabled
              className="opacity-50 cursor-not-allowed"
            >
              <Download className="mr-2 h-4 w-4" />
              Download Unavailable
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Export Filter Dialog */}
      <Dialog open={exportDialogOpen} onOpenChange={setExportDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Export {exportFormat.toUpperCase()} - Filter Options</DialogTitle>
            <p className="text-sm text-muted-foreground">
              Configure filters to export specific vehicles. Leave all unchecked to export all vehicles.
            </p>
          </DialogHeader>
          
          <div className="space-y-6 py-4">
            {/* Body Type Filter */}
            <div className="space-y-3">
              <Label className="text-base font-semibold">Body Type</Label>
              <div className="grid grid-cols-3 gap-3">
                {bodyTypeOptions.map((type) => (
                  <div key={type} className="flex items-center space-x-2">
                    <Checkbox
                      id={`export-body-${type}`}
                      checked={exportFilters.bodyTypes.includes(type)}
                      onCheckedChange={(checked) => {
                        setExportFilters(prev => ({
                          ...prev,
                          bodyTypes: checked
                            ? [...prev.bodyTypes, type]
                            : prev.bodyTypes.filter(t => t !== type)
                        }));
                      }}
                    />
                    <Label 
                      htmlFor={`export-body-${type}`}
                      className="text-sm font-normal cursor-pointer"
                    >
                      {getBodyTypeLabel(type)}
                    </Label>
                  </div>
                ))}
              </div>
            </div>

            {/* Small Vehicle Type Filter */}
            <div className="space-y-3">
              <Label className="text-base font-semibold">Small Vehicle Type</Label>
              <div className="grid grid-cols-3 gap-3">
                {smallVehicleOptions.map((type) => (
                  <div key={type} className="flex items-center space-x-2">
                    <Checkbox
                      id={`export-small-${type}`}
                      checked={exportFilters.smallVehicleTypes.includes(type)}
                      onCheckedChange={(checked) => {
                        setExportFilters(prev => ({
                          ...prev,
                          smallVehicleTypes: checked
                            ? [...prev.smallVehicleTypes, type]
                            : prev.smallVehicleTypes.filter(t => t !== type)
                        }));
                      }}
                    />
                    <Label 
                      htmlFor={`export-small-${type}`}
                      className="text-sm font-normal cursor-pointer"
                    >
                      {getSmallVehicleLabel(type)}
                    </Label>
                  </div>
                ))}
              </div>
            </div>

            {/* Axle Type Filter */}
            <div className="space-y-3">
              <Label className="text-base font-semibold">Axle Type</Label>
              <div className="grid grid-cols-3 gap-3">
                {axleTypeOptions.map((type) => (
                  <div key={type} className="flex items-center space-x-2">
                    <Checkbox
                      id={`export-axle-${type}`}
                      checked={exportFilters.axleTypes.includes(type)}
                      onCheckedChange={(checked) => {
                        setExportFilters(prev => ({
                          ...prev,
                          axleTypes: checked
                            ? [...prev.axleTypes, type]
                            : prev.axleTypes.filter(t => t !== type)
                        }));
                      }}
                    />
                    <Label 
                      htmlFor={`export-axle-${type}`}
                      className="text-sm font-normal capitalize cursor-pointer"
                    >
                      {type}
                    </Label>
                  </div>
                ))}
              </div>
            </div>

            {/* Confidence Range */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-base font-semibold">Confidence Range</Label>
                <span className="text-sm text-muted-foreground">
                  {exportFilters.confidenceRange[0]}% - {exportFilters.confidenceRange[1]}%
                </span>
              </div>
              <Slider
                value={exportFilters.confidenceRange}
                onValueChange={(value) => {
                  setExportFilters(prev => ({
                    ...prev,
                    confidenceRange: value
                  }));
                }}
                min={0}
                max={100}
                step={1}
                className="w-full"
              />
            </div>

            {/* Summary */}
            <div className="bg-muted p-4 rounded-lg space-y-2">
              <p className="text-sm font-medium">Export Summary</p>
              <p className="text-sm text-muted-foreground">
                {(() => {
                  const exportCount = jobDetails.trucks.filter((truck: any) => {
                    if (exportFilters.bodyTypes.length > 0) {
                      const matchesBodyType = truck.body_type && exportFilters.bodyTypes.includes(truck.body_type);
                      const matchesSmallVehicle = truck.small_vehicle_type && exportFilters.smallVehicleTypes.includes(truck.small_vehicle_type);
                      if (!matchesBodyType && !matchesSmallVehicle) return false;
                    }
                    if (exportFilters.smallVehicleTypes.length > 0 && exportFilters.bodyTypes.length === 0) {
                      if (!truck.small_vehicle_type || !exportFilters.smallVehicleTypes.includes(truck.small_vehicle_type)) return false;
                    }
                    if (exportFilters.axleTypes.length > 0) {
                      if (!truck.axle_type || !exportFilters.axleTypes.includes(truck.axle_type)) return false;
                    }
                    const confidence = truck.confidence * 100;
                    if (confidence < exportFilters.confidenceRange[0] || confidence > exportFilters.confidenceRange[1]) return false;
                    return true;
                  }).length;
                  
                  const hasFilters = exportFilters.bodyTypes.length > 0 ||
                                   exportFilters.smallVehicleTypes.length > 0 ||
                                   exportFilters.axleTypes.length > 0 ||
                                   exportFilters.confidenceRange[0] !== 0 ||
                                   exportFilters.confidenceRange[1] !== 100;
                  
                  return hasFilters
                    ? `${exportCount} vehicle${exportCount !== 1 ? 's' : ''} will be exported with filters: ${
                        [
                          exportFilters.bodyTypes.length > 0 && `${exportFilters.bodyTypes.length} body type(s)`,
                          exportFilters.smallVehicleTypes.length > 0 && `${exportFilters.smallVehicleTypes.length} small vehicle(s)`,
                          exportFilters.axleTypes.length > 0 && `${exportFilters.axleTypes.length} axle type(s)`,
                          (exportFilters.confidenceRange[0] !== 0 || exportFilters.confidenceRange[1] !== 100) && 
                            `${exportFilters.confidenceRange[0]}-${exportFilters.confidenceRange[1]}% confidence`
                        ].filter(Boolean).join(', ')
                      }`
                    : `${exportCount} vehicle${exportCount !== 1 ? 's' : ''} will be exported (no filters applied)`;
                })()}
              </p>
            </div>
          </div>

          <div className="flex justify-between gap-3 pt-4">
            <Button
              variant="outline"
              onClick={() => {
                setExportFilters({
                  bodyTypes: [],
                  axleTypes: [],
                  smallVehicleTypes: [],
                  confidenceRange: [0, 100],
                });
              }}
            >
              Clear Filters
            </Button>
            <div className="flex gap-3">
              <Button variant="outline" onClick={() => setExportDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleExport}>
                <Download className="mr-2 h-4 w-4" />
                Export {exportFormat.toUpperCase()}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Job?</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this job? This will permanently erase all data 
              from the server including the video file, all detected trucks, and crop images. 
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              disabled={isDeleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isDeleting ? "Deleting..." : "Delete Job"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
