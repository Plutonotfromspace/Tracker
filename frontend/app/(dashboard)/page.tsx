"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
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
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { mockJobs, mockStats, mockDashboardAnalytics, mockAllTrucks } from "@/lib/mock-data";
import { Job } from "@/lib/types";
import {
  CheckCircle2,
  Clock,
  XCircle,
  FileVideo,
  TruckIcon,
  Upload,
  Download,
  ArrowRight,
  Filter,
  X as XIcon,
  ChevronDown,
  FileDown,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Bar, BarChart, CartesianGrid, XAxis, Line, LineChart, Cell } from "recharts";

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
    hour: "numeric",
    minute: "2-digit",
  });
}

function getVehicleTypeColor(type: string): string {
  const colors: Record<string, string> = {
    "Dry Van": "#3b82f6",      // Blue
    "Reefer": "#10b981",        // Green
    "Flatbed": "#f97316",       // Orange
    "Bobtail": "#a855f7",       // Purple
    "Box Truck": "#eab308",     // Yellow
    "Pickup": "#ec4899",        // Pink
    "Van": "#06b6d4",           // Cyan
  };
  return colors[type] || "hsl(var(--chart-1))";
}

// Chart configurations
const vehicleChartConfig = {
  count: {
    label: "Vehicles",
    color: "hsl(var(--chart-1))",
  },
} satisfies ChartConfig;

const trafficChartConfig = {
  trucks: {
    label: "Trucks",
    color: "#3b82f6", // Blue color
  },
} satisfies ChartConfig;

export default function DashboardPage() {
  const router = useRouter();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [stats, setStats] = useState(mockStats);
  const [analytics, setAnalytics] = useState(mockDashboardAnalytics);
  const [isLoading, setIsLoading] = useState(true);
  const [dateMode, setDateMode] = useState<'upload' | 'video'>('upload');
  const [vehicleFilter, setVehicleFilter] = useState<string>('all');
  const [allTrucks, setAllTrucks] = useState<any[]>([]);
  const [colorMode, setColorMode] = useState(false);
  
  // Advanced filter states
  const [selectedBodyTypes, setSelectedBodyTypes] = useState<string[]>([]);
  const [selectedAxleTypes, setSelectedAxleTypes] = useState<string[]>([]);
  const [selectedSmallVehicleTypes, setSelectedSmallVehicleTypes] = useState<string[]>([]);
  const [confidenceRange, setConfidenceRange] = useState<number[]>([0, 100]);
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  
  // Export dialog states
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportFormat, setExportFormat] = useState<'csv' | 'json'>('csv');
  const [exportFilters, setExportFilters] = useState({
    bodyTypes: [] as string[],
    axleTypes: [] as string[],
    smallVehicleTypes: [] as string[],
    confidenceRange: [0, 100] as number[],
  });

  const vehicleTypes = [
    { value: 'all', label: 'All Vehicles' },
    { value: 'dry_van', label: 'Dry Van' },
    { value: 'reefer', label: 'Reefer' },
    { value: 'flatbed', label: 'Flatbed' },
    { value: 'bobtail', label: 'Bobtail' },
    { value: 'box_truck', label: 'Box Truck' },
    { value: 'pickup', label: 'Pickup' },
    { value: 'van', label: 'Van' },
  ];

  const bodyTypeOptions = ["dry_van", "reefer", "flatbed"];
  const axleTypeOptions = ["standard", "spread"];
  const smallVehicleOptions = ["bobtail", "box_truck", "pickup", "van", "other"];
  
  const getBodyTypeLabel = (bodyType: string): string => {
    const labels: Record<string, string> = {
      dry_van: "Dry Van",
      reefer: "Reefer",
      flatbed: "Flatbed",
    };
    return labels[bodyType] || bodyType;
  };

  const getSmallVehicleLabel = (vehicleType: string): string => {
    const labels: Record<string, string> = {
      bobtail: "Bobtail",
      box_truck: "Box Truck",
      pickup: "Pickup",
      van: "Van",
      other: "Other",
    };
    return labels[vehicleType] || vehicleType;
  };

  const activeFiltersCount =
    selectedBodyTypes.length +
    selectedAxleTypes.length +
    selectedSmallVehicleTypes.length +
    (confidenceRange[0] !== 0 || confidenceRange[1] !== 100 ? 1 : 0);

  const clearAllFilters = () => {
    setSelectedBodyTypes([]);
    setSelectedAxleTypes([]);
    setSelectedSmallVehicleTypes([]);
    setConfidenceRange([0, 100]);
  };

  const handleExport = () => {
    // Filter trucks based on export filters
    const trucksToExport = allTrucks.filter((truck: any) => {
      // Body type filter
      const hasBodyTypeFilter = exportFilters.bodyTypes.length > 0;
      const hasSmallVehicleFilter = exportFilters.smallVehicleTypes.length > 0;
      
      if (hasBodyTypeFilter || hasSmallVehicleFilter) {
        const matchesBodyType = hasBodyTypeFilter && truck.body_type && exportFilters.bodyTypes.includes(truck.body_type);
        const matchesSmallVehicle = hasSmallVehicleFilter && truck.small_vehicle_type && exportFilters.smallVehicleTypes.includes(truck.small_vehicle_type);
        
        if (!matchesBodyType && !matchesSmallVehicle) {
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
      const headers = ['Vehicle ID', 'Job ID', 'Timestamp', 'Body Type', 'Axle Type', 'Small Vehicle Type', 'Confidence'];
      const rows = trucksToExport.map((truck: any) => [
        truck.unique_truck_id || truck.truck_id,
        truck.job_id,
        truck.timestamp,
        truck.body_type || '',
        truck.axle_type || '',
        truck.small_vehicle_type || '',
        `${(truck.confidence * 100).toFixed(1)}%`,
      ]);
      
      const csvContent = [
        headers.join(','),
        ...rows.map((row: string[]) => row.map((cell: string) => `"${cell}"`).join(','))
      ].join('\n');
      
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'master_data_trucks.csv';
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
      a.download = 'master_data_trucks.json';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    }
    
    setExportDialogOpen(false);
  };

  useEffect(() => {
    const fetchData = async () => {
      const dataSource = localStorage.getItem("data_source");
      
      if (dataSource === "mock") {
        setJobs(mockJobs);
        setStats(mockStats);
        setAnalytics(mockDashboardAnalytics);
        setAllTrucks(mockAllTrucks);
        setIsLoading(false);
        return;
      }

      // Fetch live data
      try {
        const token = localStorage.getItem("auth_token");
        const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        
        const response = await fetch(`${API_URL}/api/jobs/`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const data = await response.json();
          setJobs(data);
          
          // Calculate stats from jobs
          const calculatedStats = {
            total_jobs: data.length,
            processing: data.filter((j: Job) => j.status === "processing").length,
            completed: data.filter((j: Job) => j.status === "completed").length,
            failed: data.filter((j: Job) => j.status === "failed").length,
            total_trucks: data.reduce((sum: number, j: Job) => sum + (j.truck_count || 0), 0),
          };
          setStats(calculatedStats);
          
          // Fetch trucks data to calculate analytics
          const trucksResponse = await fetch(`${API_URL}/api/trucks/`, {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          });
          
          if (trucksResponse.ok) {
            const trucksData = await trucksResponse.json();
            setAllTrucks(trucksData);
            
            // Calculate vehicle type distribution
            const distribution: Record<string, number> = {
              dry_van: 0,
              reefer: 0,
              flatbed: 0,
              bobtail: 0,
              box_truck: 0,
              pickup: 0,
              van: 0,
            };
            
            trucksData.forEach((truck: any) => {
              if (truck.body_type && distribution.hasOwnProperty(truck.body_type)) {
                distribution[truck.body_type]++;
              }
              if (truck.small_vehicle_type && distribution.hasOwnProperty(truck.small_vehicle_type)) {
                distribution[truck.small_vehicle_type]++;
              }
            });
            
            // Calculate traffic trends from completed jobs
            const completedJobs = data.filter((j: Job) => j.status === "completed" && j.completed_at);
            const trafficTrends = [];
            
            if (completedJobs.length > 0) {
              const allDates = completedJobs.map((job: Job) => new Date(job.completed_at!));
              const earliestDate = new Date(Math.min(...allDates.map((d: Date) => d.getTime())));
              const latestDate = new Date(Math.max(...allDates.map((d: Date) => d.getTime())));
              
              const startDate = new Date(earliestDate);
              startDate.setHours(0, 0, 0, 0);
              
              const endDate = new Date(latestDate);
              endDate.setHours(23, 59, 59, 999);
              
              const currentDate = new Date(startDate);
              while (currentDate <= endDate) {
                const dateStr = currentDate.toISOString().split('T')[0];
                
                const jobsOnDay = completedJobs.filter((job: Job) => {
                  const jobDate = new Date(job.completed_at!).toISOString().split('T')[0];
                  return jobDate === dateStr;
                });
                
                const trucksOnDay = jobsOnDay.reduce((sum: number, job: Job) => sum + (job.truck_count || 0), 0);
                
                trafficTrends.push({
                  date: dateStr,
                  trucks: trucksOnDay,
                });
                
                currentDate.setDate(currentDate.getDate() + 1);
              }
            }
            
            setAnalytics({
              vehicleTypeDistribution: {
                dry_van: distribution.dry_van,
                reefer: distribution.reefer,
                flatbed: distribution.flatbed,
                bobtail: distribution.bobtail,
                box_truck: distribution.box_truck,
                pickup: distribution.pickup,
                van: distribution.van,
              },
              trafficTrends,
            });
          }
        }
      } catch (error) {
        console.error("Failed to fetch jobs:", error);
      } finally {
        setIsLoading(false);
      }
    };

    // Initial fetch
    fetchData();

    // Poll for updates every 3 seconds
    const pollInterval = setInterval(() => {
      const dataSource = localStorage.getItem("data_source");
      if (dataSource !== "mock") {
        fetchData();
      }
    }, 3000);

    return () => clearInterval(pollInterval);
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  // Extract video date from job ID (format: YYYYMMDD_HHMMSS_xxx or similar)
  const getVideoDate = (job: Job): Date | null => {
    try {
      // Try to parse date from job ID (e.g., "20251207_143022_a4f1")
      const match = job.id.match(/^(\d{8})_(\d{6})/);
      if (match) {
        const dateStr = match[1]; // YYYYMMDD
        const timeStr = match[2]; // HHMMSS
        const year = parseInt(dateStr.substring(0, 4));
        const month = parseInt(dateStr.substring(4, 6)) - 1;
        const day = parseInt(dateStr.substring(6, 8));
        const hour = parseInt(timeStr.substring(0, 2));
        const minute = parseInt(timeStr.substring(2, 4));
        const second = parseInt(timeStr.substring(4, 6));
        return new Date(year, month, day, hour, minute, second);
      }
      return null;
    } catch {
      return null;
    }
  };

  // Recalculate traffic trends based on date mode and vehicle filter
  const getTrafficTrendsForMode = () => {
    const completedJobs = jobs.filter(j => j.status === "completed" && (dateMode === 'upload' ? j.completed_at : getVideoDate(j)));
    const trafficTrends = [];
    
    if (completedJobs.length > 0) {
      const allDates = completedJobs
        .map(job => dateMode === 'upload' && job.completed_at 
          ? new Date(job.completed_at) 
          : getVideoDate(job))
        .filter((date): date is Date => date !== null);
      
      if (allDates.length > 0) {
        const earliestDate = new Date(Math.min(...allDates.map((d: Date) => d.getTime())));
        const latestDate = new Date(Math.max(...allDates.map((d: Date) => d.getTime())));
        
        const startDate = new Date(earliestDate);
        startDate.setHours(0, 0, 0, 0);
        
        const endDate = new Date(latestDate);
        endDate.setHours(23, 59, 59, 999);
        
        const currentDate = new Date(startDate);
        while (currentDate <= endDate) {
          const dateStr = currentDate.toISOString().split('T')[0];
          
          const jobsOnDay = completedJobs.filter(job => {
            const jobDate = dateMode === 'upload' && job.completed_at
              ? new Date(job.completed_at)
              : getVideoDate(job);
            if (!jobDate) return false;
            const jobDateStr = jobDate.toISOString().split('T')[0];
            return jobDateStr === dateStr;
          });
          
          let trucksOnDay = 0;
          
          if (vehicleFilter === 'all') {
            // Count all trucks
            trucksOnDay = jobsOnDay.reduce((sum, job) => sum + (job.truck_count || 0), 0);
          } else {
            // Filter trucks by vehicle type
            const jobIds = jobsOnDay.map(j => j.id);
            const filteredTrucks = allTrucks.filter(truck => 
              jobIds.includes(truck.job_id) &&
              (truck.body_type === vehicleFilter || truck.small_vehicle_type === vehicleFilter)
            );
            trucksOnDay = filteredTrucks.length;
          }
          
          trafficTrends.push({
            date: dateStr,
            trucks: trucksOnDay,
          });
          
          currentDate.setDate(currentDate.getDate() + 1);
        }
      }
    }
    
    return trafficTrends;
  };

  const trafficTrendsData = getTrafficTrendsForMode();

  // Get the current vehicle filter label and color
  const vehicleFilterLabel = vehicleTypes.find(v => v.value === vehicleFilter)?.label || 'Vehicles';
  const lineColor = colorMode && vehicleFilter !== 'all' 
    ? getVehicleTypeColor(vehicleFilterLabel)
    : "#3b82f6";

  // Calculate vehicle type percentages for the 5th card
  const totalVehicles = Object.values(analytics.vehicleTypeDistribution).reduce((sum, val) => sum + val, 0);
  const topVehicleType = totalVehicles > 0 
    ? Object.entries(analytics.vehicleTypeDistribution)
        .sort(([,a], [,b]) => b - a)[0]
    : null;
  const topVehiclePercent = topVehicleType && totalVehicles > 0 
    ? Math.round((topVehicleType[1] / totalVehicles) * 100)
    : 0;

  // Filter trucks based on advanced filters
  const filteredTrucks = allTrucks.filter((truck: any) => {
    // Body type and small vehicle type filter
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

    return true;
  });

  // Calculate vehicle type distribution from filtered trucks
  const filteredVehicleDistribution: Record<string, number> = {};
  filteredTrucks.forEach((truck: any) => {
    let type: string;
    if (truck.body_type) {
      type = truck.body_type;
    } else if (truck.small_vehicle_type) {
      type = truck.small_vehicle_type;
    } else {
      type = 'other';
    }
    filteredVehicleDistribution[type] = (filteredVehicleDistribution[type] || 0) + 1;
  });

  // Prepare chart data
  const vehicleChartData = Object.entries(
    activeFiltersCount > 0 ? filteredVehicleDistribution : analytics.vehicleTypeDistribution
  )
    .map(([type, count]) => ({
      type: type.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' '),
      count,
      fill: `hsl(var(--chart-${Object.keys(analytics.vehicleTypeDistribution).indexOf(type) + 1}))`,
    }))
    .sort((a, b) => b.count - a.count);

  const recentJobs = jobs.slice(0, 5);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Analytics and insights across all processing jobs
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Jobs</CardTitle>
            <FileVideo className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.total_jobs}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Processing</CardTitle>
            <Clock className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.processing}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Completed</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.completed}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Failed</CardTitle>
            <XCircle className="h-4 w-4 text-red-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.failed}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Top Vehicle Type</CardTitle>
            <TruckIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{topVehiclePercent}%</div>
            <p className="text-xs text-muted-foreground mt-1">
              {topVehicleType ? topVehicleType[0].split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ') : 'No data'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <div className="flex gap-3 flex-wrap">
        <Button size="lg" onClick={() => router.push('/upload')}>
          <Upload className="mr-2 h-5 w-5" />
          Upload New Video
        </Button>
        <Button size="lg" variant="outline" onClick={() => {
          setExportFormat('csv');
          setExportFilters({
            bodyTypes: selectedBodyTypes,
            axleTypes: selectedAxleTypes,
            smallVehicleTypes: selectedSmallVehicleTypes,
            confidenceRange: confidenceRange,
          });
          setExportDialogOpen(true);
        }}>
          <FileDown className="mr-2 h-5 w-5" />
          Export CSV
        </Button>
        <Button size="lg" variant="outline" onClick={() => {
          setExportFormat('json');
          setExportFilters({
            bodyTypes: selectedBodyTypes,
            axleTypes: selectedAxleTypes,
            smallVehicleTypes: selectedSmallVehicleTypes,
            confidenceRange: confidenceRange,
          });
          setExportDialogOpen(true);
        }}>
          <FileDown className="mr-2 h-5 w-5" />
          Export JSON
        </Button>
        <Button size="lg" variant="ghost" onClick={() => router.push('/jobs')}>
          View All Jobs
          <ArrowRight className="ml-2 h-5 w-5" />
        </Button>
      </div>

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

                {/* Small Vehicle Type Filter */}
                <div className="space-y-3 md:col-span-2">
                  <Label className="text-sm font-medium">Small Vehicle Type</Label>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-5">
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
                  <Label className="text-sm font-medium">Confidence Range</Label>
                  <span className="text-sm text-muted-foreground">
                    {confidenceRange[0]}% - {confidenceRange[1]}%
                  </span>
                </div>
                <Slider
                  min={0}
                  max={100}
                  step={1}
                  value={confidenceRange}
                  onValueChange={setConfidenceRange}
                  className="w-full"
                />
              </div>
            </CardContent>
          </CollapsibleContent>
        </Collapsible>
      </Card>

      {/* Analytics Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Vehicle Type Distribution */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Vehicle Type Distribution</CardTitle>
                <CardDescription>
                  {activeFiltersCount > 0 
                    ? `Showing ${filteredTrucks.length.toLocaleString()} of ${stats.total_trucks.toLocaleString()} vehicles (filtered)`
                    : `Breakdown across all ${stats.total_trucks.toLocaleString()} vehicles detected`
                  }
                </CardDescription>
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
            <ChartContainer config={vehicleChartConfig} className="aspect-auto h-[250px] w-full sm:h-[300px]">
              <BarChart data={vehicleChartData}>
                <CartesianGrid vertical={false} />
                <XAxis 
                  dataKey="type" 
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                  fontSize={11}
                />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar 
                  dataKey="count" 
                  radius={[4, 4, 0, 0]}
                >
                  {vehicleChartData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={colorMode ? getVehicleTypeColor(entry.type) : "hsl(var(--chart-1))"} 
                    />
                  ))}
                </Bar>
              </BarChart>
            </ChartContainer>
          </CardContent>
        </Card>

        {/* Traffic Trends */}
        <Card>
          <CardHeader>
            <div className="flex flex-col gap-4">
              <div>
                <CardTitle>Traffic Trends</CardTitle>
                <CardDescription>
                  {trafficTrendsData.length > 0 
                    ? `${vehicleTypes.find(v => v.value === vehicleFilter)?.label} counted per day from ${new Date(trafficTrendsData[0].date).toLocaleDateString("en-US", { month: "short", day: "numeric" })} to ${new Date(trafficTrendsData[trafficTrendsData.length - 1].date).toLocaleDateString("en-US", { month: "short", day: "numeric" })}`
                    : "No traffic data available"}
                </CardDescription>
              </div>
              <div className="flex flex-col sm:flex-row gap-2">
                <Select value={vehicleFilter} onValueChange={setVehicleFilter}>
                  <SelectTrigger className="w-full sm:w-40 h-9">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {vehicleTypes.map((type) => (
                      <SelectItem key={type.value} value={type.value}>
                        {type.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <div className="flex gap-1 bg-muted p-1 rounded-md">
                  <Button
                    variant={dateMode === 'upload' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setDateMode('upload')}
                    className="h-7 text-xs flex-1 sm:flex-none"
                  >
                    Upload Date
                  </Button>
                  <Button
                    variant={dateMode === 'video' ? 'default' : 'ghost'}
                    size="sm"
                  onClick={() => setDateMode('video')}
                  className="h-7 text-xs flex-1 sm:flex-none"
                >
                  Video Date
                </Button>
                </div>
              </div>
            </div>
          </CardHeader>
          <CardContent className="px-2 sm:p-6">
            {(() => {
              const minDaysRequired = 7;
              const daysOfData = trafficTrendsData.length;
              
              if (daysOfData < minDaysRequired) {
                return (
                  <div className="flex flex-col items-center justify-center h-[300px] text-center px-8">
                    <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
                      <TruckIcon className="w-8 h-8 text-muted-foreground" />
                    </div>
                    <h3 className="text-lg font-medium mb-2">Gathering Data</h3>
                    <p className="text-sm text-muted-foreground max-w-md">
                      Traffic trends require at least {minDaysRequired} days of data to display meaningful patterns.
                      You currently have {daysOfData} day{daysOfData !== 1 ? 's' : ''} of data.
                    </p>
                    <p className="text-sm text-muted-foreground mt-2">
                      {minDaysRequired - daysOfData} more day{minDaysRequired - daysOfData !== 1 ? 's' : ''} needed to unlock this view.
                    </p>
                  </div>
                );
              }
              
              return (
                <ChartContainer config={trafficChartConfig} className="aspect-auto h-[250px] w-full sm:h-[300px]">
                  <LineChart data={trafficTrendsData}>
                    <CartesianGrid vertical={false} />
                    <XAxis 
                      dataKey="date" 
                      tickLine={false}
                      axisLine={false}
                      tickMargin={8}
                      tickFormatter={(value) => {
                        // Parse date string as local date to avoid timezone issues
                        const [year, month, day] = value.split('-').map(Number);
                        const date = new Date(year, month - 1, day);
                        return date.toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                        });
                      }}
                      minTickGap={30}
                    />
                    <ChartTooltip 
                      content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          // Parse date string as local date to avoid timezone issues
                          const [year, month, day] = label.split('-').map(Number);
                          const date = new Date(year, month - 1, day);
                          const formattedDate = date.toLocaleDateString("en-US", {
                            month: "long",
                            day: "numeric",
                            year: "numeric",
                          });
                          
                          return (
                            <div className="rounded-lg border bg-background p-2 shadow-sm">
                              <div className="text-xs text-muted-foreground mb-1">
                                {formattedDate}
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="h-2 w-2 rounded-full" style={{ backgroundColor: lineColor }} />
                                <span className="text-sm font-medium">
                                  {payload[0].value} {vehicleFilterLabel}
                                </span>
                              </div>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="trucks" 
                      stroke={lineColor}
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ChartContainer>
              );
            })()}
          </CardContent>
        </Card>
      </div>

      {/* Recent Jobs */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>Last 5 processing jobs</CardDescription>
          </div>
          <Link href="/jobs">
            <Button variant="ghost" size="sm">
              View All
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </Link>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Video Name</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Trucks</TableHead>
                <TableHead>Completed</TableHead>
                <TableHead></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {recentJobs.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} className="text-center text-muted-foreground py-8">
                    No jobs found. Upload a video to get started.
                  </TableCell>
                </TableRow>
              ) : (
                recentJobs.map((job) => (
                  <TableRow 
                    key={job.id} 
                    className="cursor-pointer hover:bg-muted/50"
                    onClick={() => {
                      if (job.status !== "failed") {
                        router.push(`/jobs/${job.id}`);
                      }
                    }}
                  >
                    <TableCell className="font-medium max-w-xs truncate">
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
                      {formatDate(job.completed_at)}
                    </TableCell>
                    <TableCell>
                      {job.status !== "failed" && (
                        <Button variant="ghost" size="sm">
                          View
                          <ArrowRight className="ml-2 h-4 w-4" />
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Export Dialog */}
      <Dialog open={exportDialogOpen} onOpenChange={setExportDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              Export {exportFormat.toUpperCase()} - Filter Options
            </DialogTitle>
            <p className="text-sm text-muted-foreground">
              Configure filters to export specific vehicles. Leave all unchecked to export all vehicles.
            </p>
          </DialogHeader>
          
          <div className="space-y-6 py-4">
            {/* Body Type */}
            <div className="space-y-3">
              <Label className="text-base font-semibold">Body Type</Label>
              <div className="grid grid-cols-3 gap-3">
                {bodyTypeOptions.map((type) => (
                  <div key={type} className="flex items-center space-x-2">
                    <Checkbox
                      id={`export-body-${type}`}
                      checked={exportFilters.bodyTypes.includes(type)}
                      onCheckedChange={(checked) => {
                        if (checked) {
                          setExportFilters({
                            ...exportFilters,
                            bodyTypes: [...exportFilters.bodyTypes, type],
                          });
                        } else {
                          setExportFilters({
                            ...exportFilters,
                            bodyTypes: exportFilters.bodyTypes.filter((t) => t !== type),
                          });
                        }
                      }}
                    />
                    <label
                      htmlFor={`export-body-${type}`}
                      className="text-sm cursor-pointer"
                    >
                      {getBodyTypeLabel(type)}
                    </label>
                  </div>
                ))}
              </div>
            </div>

            {/* Small Vehicle Type */}
            <div className="space-y-3">
              <Label className="text-base font-semibold">Small Vehicle Type</Label>
              <div className="grid grid-cols-3 gap-3">
                {smallVehicleOptions.map((type) => (
                  <div key={type} className="flex items-center space-x-2">
                    <Checkbox
                      id={`export-small-${type}`}
                      checked={exportFilters.smallVehicleTypes.includes(type)}
                      onCheckedChange={(checked) => {
                        if (checked) {
                          setExportFilters({
                            ...exportFilters,
                            smallVehicleTypes: [...exportFilters.smallVehicleTypes, type],
                          });
                        } else {
                          setExportFilters({
                            ...exportFilters,
                            smallVehicleTypes: exportFilters.smallVehicleTypes.filter((t) => t !== type),
                          });
                        }
                      }}
                    />
                    <label
                      htmlFor={`export-small-${type}`}
                      className="text-sm cursor-pointer"
                    >
                      {getSmallVehicleLabel(type)}
                    </label>
                  </div>
                ))}
              </div>
            </div>

            {/* Axle Type */}
            <div className="space-y-3">
              <Label className="text-base font-semibold">Axle Type</Label>
              <div className="grid grid-cols-3 gap-3">
                {axleTypeOptions.map((type) => (
                  <div key={type} className="flex items-center space-x-2">
                    <Checkbox
                      id={`export-axle-${type}`}
                      checked={exportFilters.axleTypes.includes(type)}
                      onCheckedChange={(checked) => {
                        if (checked) {
                          setExportFilters({
                            ...exportFilters,
                            axleTypes: [...exportFilters.axleTypes, type],
                          });
                        } else {
                          setExportFilters({
                            ...exportFilters,
                            axleTypes: exportFilters.axleTypes.filter((t) => t !== type),
                          });
                        }
                      }}
                    />
                    <label
                      htmlFor={`export-axle-${type}`}
                      className="text-sm cursor-pointer capitalize"
                    >
                      {type}
                    </label>
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
                min={0}
                max={100}
                step={1}
                value={exportFilters.confidenceRange}
                onValueChange={(value) => {
                  setExportFilters({
                    ...exportFilters,
                    confidenceRange: value,
                  });
                }}
                className="w-full"
              />
            </div>

            {/* Export Summary */}
            <div className="rounded-lg bg-muted p-4">
              <p className="text-sm font-medium mb-2">Export Summary</p>
              <p className="text-sm text-muted-foreground">
                {(() => {
                  const exportCount = allTrucks.filter((truck: any) => {
                    const hasBodyTypeFilter = exportFilters.bodyTypes.length > 0;
                    const hasSmallVehicleFilter = exportFilters.smallVehicleTypes.length > 0;
                    
                    if (hasBodyTypeFilter || hasSmallVehicleFilter) {
                      const matchesBodyType = hasBodyTypeFilter && truck.body_type && exportFilters.bodyTypes.includes(truck.body_type);
                      const matchesSmallVehicle = hasSmallVehicleFilter && truck.small_vehicle_type && exportFilters.smallVehicleTypes.includes(truck.small_vehicle_type);
                      
                      if (!matchesBodyType && !matchesSmallVehicle) {
                        return false;
                      }
                    }

                    if (exportFilters.axleTypes.length > 0) {
                      if (!truck.axle_type || !exportFilters.axleTypes.includes(truck.axle_type)) {
                        return false;
                      }
                    }

                    const confidence = truck.confidence * 100;
                    if (confidence < exportFilters.confidenceRange[0] || confidence > exportFilters.confidenceRange[1]) {
                      return false;
                    }

                    return true;
                  }).length;

                  const hasFilters = exportFilters.bodyTypes.length > 0 ||
                                   exportFilters.smallVehicleTypes.length > 0 ||
                                   exportFilters.axleTypes.length > 0 ||
                                   exportFilters.confidenceRange[0] !== 0 ||
                                   exportFilters.confidenceRange[1] !== 100;

                  if (!hasFilters) {
                    return `${exportCount} vehicle${exportCount !== 1 ? 's' : ''} will be exported (no filters applied)`;
                  } else {
                    const filterDesc = [
                      exportFilters.bodyTypes.length > 0 && `${exportFilters.bodyTypes.length} body type(s)`,
                      exportFilters.smallVehicleTypes.length > 0 && `${exportFilters.smallVehicleTypes.length} small vehicle(s)`,
                      exportFilters.axleTypes.length > 0 && `${exportFilters.axleTypes.length} axle type(s)`,
                      (exportFilters.confidenceRange[0] !== 0 || exportFilters.confidenceRange[1] !== 100) && 
                        `${exportFilters.confidenceRange[0]}-${exportFilters.confidenceRange[1]}% confidence`
                    ].filter(Boolean).join(', ');
                    return `${exportCount} vehicle${exportCount !== 1 ? 's' : ''} will be exported with filters: ${filterDesc}`;
                  }
                })()}
              </p>
            </div>
          </div>

          <div className="flex items-center justify-between pt-4 border-t">
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
            <div className="flex gap-2">
              <Button variant="ghost" onClick={() => setExportDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleExport}>
                <FileDown className="mr-2 h-4 w-4" />
                Export {exportFormat.toUpperCase()}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
