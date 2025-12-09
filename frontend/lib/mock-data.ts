import { Job, DashboardStats, JobDetails, Truck } from "./types";

/**
 * Mock data for development
 */

const generateRandomTruckCount = () => Math.floor(Math.random() * (500 - 50 + 1)) + 50;

const calculateProcessingTime = (truckCount: number): number => {
  // Base time: 20 minutes (1200 seconds)
  // Max time: 70 minutes (4200 seconds) 
  // Scale based on truck count (50-500 trucks)
  const minTime = 20 * 60 * 1000; // 20 minutes in ms
  const maxTime = 70 * 60 * 1000; // 70 minutes in ms
  const truckRange = 500 - 50;
  const timeRange = maxTime - minTime;
  const ratio = (truckCount - 50) / truckRange;
  return Math.floor(minTime + (timeRange * ratio));
};

const generateJobDates = (daysAgo: number, truckCount?: number) => {
  const baseDate = new Date();
  baseDate.setDate(baseDate.getDate() - daysAgo);
  baseDate.setHours(Math.floor(Math.random() * 24));
  baseDate.setMinutes(Math.floor(Math.random() * 60));
  
  const created_at = baseDate.toISOString();
  const started_at = new Date(baseDate.getTime() + 3000).toISOString(); // Start 3 seconds after creation
  
  if (truckCount) {
    const processingTime = calculateProcessingTime(truckCount);
    const completed_at = new Date(baseDate.getTime() + 3000 + processingTime).toISOString();
    return { created_at, started_at, completed_at };
  }
  
  return { created_at, started_at };
};

// Generate completed job with proper dates
const createCompletedJob = (id: string, videoName: string, videoPath: string, daysAgo: number): Job => {
  const truckCount = generateRandomTruckCount();
  const dates = generateJobDates(daysAgo, truckCount);
  return {
    id,
    video_name: videoName,
    video_path: videoPath,
    status: "completed",
    ...dates,
    truck_count: truckCount,
  };
};

export const mockJobs: Job[] = [
  // Active jobs (newest first) - uploaded today but videos from October
  {
    id: "20251015_185000_o8t5",
    video_name: "Border Analysis Oct 15th.mp4",
    video_path: "/videos/analysis_oct15.mp4",
    status: "created",
    created_at: new Date(Date.now() - 2 * 60 * 1000).toISOString(), // 2 minutes ago
  },
  {
    id: "20251018_180000_n7s4",
    video_name: "Peak Hours Oct 18th.mp4",
    video_path: "/videos/peak_oct18.mp4",
    status: "processing",
    created_at: new Date(Date.now() - 10 * 60 * 1000).toISOString(), // 10 minutes ago
    started_at: new Date(Date.now() - 8 * 60 * 1000).toISOString(), // 8 minutes ago
    truck_count: 0,
  },
  // Completed jobs - uploaded over last 2 weeks, but videos from mid-October to early November
  createCompletedJob(
    "20251105_143022_a4f1",
    "World Trade Bridge Nov 5th Morning.mp4",
    "/videos/wtb_nov5_morning.mp4",
    0
  ),
  createCompletedJob(
    "20251102_143022_b5g2",
    "World Trade Bridge Nov 2nd Afternoon.mp4",
    "/videos/wtb_nov2_afternoon.mp4",
    1
  ),
  createCompletedJob(
    "20251030_151544_c6h3",
    "Morning Rush Hour Oct 30th.mp4",
    "/videos/morning_oct30.mp4",
    2
  ),
  createCompletedJob(
    "20251028_160812_d7i4",
    "Evening Traffic Oct 28th.mp4",
    "/videos/evening_oct28.mp4",
    3
  ),
  createCompletedJob(
    "20251025_162234_e8j5",
    "Border Crossing Oct 25th Peak Hours.mp4",
    "/videos/border_oct25.mp4",
    4
  ),
  createCompletedJob(
    "20251023_093015_f9k6",
    "Weekend Traffic Oct 23rd.mp4",
    "/videos/weekend_oct23.mp4",
    5
  ),
  createCompletedJob(
    "20251020_104520_g0l7",
    "Midday Operations Oct 20th.mp4",
    "/videos/midday_oct20.mp4",
    6
  ),
  createCompletedJob(
    "20251018_115630_h1m8",
    "Early Morning Oct 18th.mp4",
    "/videos/early_oct18.mp4",
    7
  ),
  createCompletedJob(
    "20251016_120740_i2n9",
    "Late Afternoon Oct 16th.mp4",
    "/videos/late_oct16.mp4",
    8
  ),
  createCompletedJob(
    "20251014_131850_j3o0",
    "Rush Hour Oct 14th.mp4",
    "/videos/rush_oct14.mp4",
    9
  ),
  createCompletedJob(
    "20251012_142960_k4p1",
    "Evening Shift Oct 12th.mp4",
    "/videos/evening_oct12.mp4",
    10
  ),
  createCompletedJob(
    "20251010_154070_l5q2",
    "Morning Operations Oct 10th.mp4",
    "/videos/morning_oct10.mp4",
    11
  ),
  createCompletedJob(
    "20251008_165180_m6r3",
    "Afternoon Traffic Oct 8th.mp4",
    "/videos/afternoon_oct8.mp4",
    12
  ),
];

export const mockStats: DashboardStats = {
  total_jobs: mockJobs.length,
  processing: mockJobs.filter((j) => j.status === "processing").length,
  completed: mockJobs.filter((j) => j.status === "completed").length,
  failed: mockJobs.filter((j) => j.status === "failed").length,
  total_trucks: mockJobs.reduce((acc, job) => acc + (job.truck_count || 0), 0),
};

// Mock detailed job data with trucks
const generateMockTrucks = (jobId: string, count: number): Truck[] => {
  const bodyTypes: Array<"dry_van" | "reefer" | "flatbed"> = [
    "dry_van",
    "reefer",
    "flatbed",
  ];
  const axleTypes: Array<"standard" | "spread"> = ["standard", "spread"];
  const smallVehicleTypes: Array<"bobtail" | "box_truck" | "pickup" | "van" | "other"> = [
    "bobtail",
    "box_truck",
    "pickup",
    "van",
    "other",
  ];

  // Extract date from jobId (format: YYYYMMDD_HHMMSS_xxx)
  const getJobDate = (jobId: string): Date => {
    const match = jobId.match(/^(\d{8})_(\d{6})/);
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
    return new Date();
  };

  const jobDate = getJobDate(jobId);

  return Array.from({ length: count }, (_, i) => {
    const truckNum = String(i + 1).padStart(3, "0");
    const isSmallVehicle = Math.random() < 0.15; // 15% chance of small vehicle
    
    // Spread truck timestamps over ~30 minutes from the job date
    const timestamp = new Date(
      jobDate.getTime() + Math.random() * 30 * 60 * 1000
    ).toISOString();

    const truck: Truck = {
      truck_id: `Truck${truckNum}`,
      unique_truck_id: `${jobId}_Truck${truckNum}`,
      job_id: jobId,
      timestamp,
      files: {
        frame: `debug_images/Truck${truckNum}/entry_frame.jpg`,
        crop: `debug_images/Truck${truckNum}/entry_crop.jpg`,
      },
      confidence: 0.85 + Math.random() * 0.14,
    };

    if (isSmallVehicle) {
      truck.small_vehicle_type = smallVehicleTypes[Math.floor(Math.random() * smallVehicleTypes.length)];
    } else {
      truck.body_type = bodyTypes[Math.floor(Math.random() * bodyTypes.length)];
      truck.axle_type = axleTypes[Math.floor(Math.random() * axleTypes.length)];
    }

    return truck;
  });
};

const generateBodyTypeDistribution = (totalTrucks: number) => {
  const smallVehicleCount = Math.floor(totalTrucks * 0.15);
  const regularTrucks = totalTrucks - smallVehicleCount;
  
  const dryVan = Math.floor(regularTrucks * 0.45);
  const reefer = Math.floor(regularTrucks * 0.38);
  const flatbed = regularTrucks - dryVan - reefer;
  
  return { dry_van: dryVan, reefer, flatbed };
};

const generateSmallVehicleDistribution = (totalTrucks: number) => {
  const smallVehicleCount = Math.floor(totalTrucks * 0.15);
  const bobtail = Math.floor(smallVehicleCount * 0.4);
  const boxTruck = Math.floor(smallVehicleCount * 0.3);
  const pickup = Math.floor(smallVehicleCount * 0.2);
  const van = smallVehicleCount - bobtail - boxTruck - pickup;
  
  return { bobtail, box_truck: boxTruck, pickup, van };
};

export const mockJobDetails: Record<string, JobDetails> = Object.fromEntries(
  mockJobs.map((job) => {
    if (job.status === "completed" && job.truck_count) {
      return [
        job.id,
        {
          ...job,
          trucks: generateMockTrucks(job.id, job.truck_count),
          body_type_distribution: generateBodyTypeDistribution(job.truck_count),
          small_vehicle_distribution: generateSmallVehicleDistribution(job.truck_count),
        },
      ];
    } else {
      return [
        job.id,
        {
          ...job,
          trucks: [],
          body_type_distribution: {},
          small_vehicle_distribution: {},
        },
      ];
    }
  })
);

// Dashboard Analytics - Aggregated data across all jobs
const generateDashboardAnalytics = () => {
  const completedJobs = mockJobs.filter(j => j.status === "completed" && j.truck_count);
  
  // Aggregate vehicle type distribution across all jobs
  const totalVehicleDistribution = {
    dry_van: 0,
    reefer: 0,
    flatbed: 0,
    bobtail: 0,
    box_truck: 0,
    pickup: 0,
    van: 0,
  };
  
  completedJobs.forEach(job => {
    const bodyDist = generateBodyTypeDistribution(job.truck_count!);
    const smallDist = generateSmallVehicleDistribution(job.truck_count!);
    
    totalVehicleDistribution.dry_van += bodyDist.dry_van;
    totalVehicleDistribution.reefer += bodyDist.reefer;
    totalVehicleDistribution.flatbed += bodyDist.flatbed;
    totalVehicleDistribution.bobtail += smallDist.bobtail;
    totalVehicleDistribution.box_truck += smallDist.box_truck;
    totalVehicleDistribution.pickup += smallDist.pickup;
    totalVehicleDistribution.van += smallDist.van;
  });
  
  // Generate traffic trends - smart date range based on actual job dates
  const trafficTrends = [];
  
  if (completedJobs.length > 0) {
    // Find earliest and latest job dates
    const allDates = completedJobs
      .filter(job => job.completed_at)
      .map(job => new Date(job.completed_at!));
    
    if (allDates.length > 0) {
      const earliestDate = new Date(Math.min(...allDates.map(d => d.getTime())));
      const latestDate = new Date(Math.max(...allDates.map(d => d.getTime())));
      
      // Start from the earliest job date
      const startDate = new Date(earliestDate);
      startDate.setHours(0, 0, 0, 0);
      
      // End at the latest job date
      const endDate = new Date(latestDate);
      endDate.setHours(23, 59, 59, 999);
      
      // Generate daily data points from start to end
      const currentDate = new Date(startDate);
      while (currentDate <= endDate) {
        const dateStr = currentDate.toISOString().split('T')[0];
        
        // Find jobs completed on this day
        const jobsOnDay = completedJobs.filter(job => {
          if (!job.completed_at) return false;
          const jobDate = new Date(job.completed_at).toISOString().split('T')[0];
          return jobDate === dateStr;
        });
        
        const trucksOnDay = jobsOnDay.reduce((sum, job) => sum + (job.truck_count || 0), 0);
        
        trafficTrends.push({
          date: dateStr,
          trucks: trucksOnDay,
        });
        
        // Move to next day
        currentDate.setDate(currentDate.getDate() + 1);
      }
    }
  }
  
  return {
    vehicleTypeDistribution: totalVehicleDistribution,
    trafficTrends,
  };
};

export const mockDashboardAnalytics = generateDashboardAnalytics();

// Export all trucks for filtering (computed after mockJobDetails is fully created)
export const mockAllTrucks = Object.values(mockJobDetails)
  .filter(jobDetail => jobDetail.status === "completed" && jobDetail.trucks)
  .flatMap(jobDetail => jobDetail.trucks);
