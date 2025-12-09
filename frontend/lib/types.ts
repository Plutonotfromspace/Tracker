/**
 * Type definitions for the Truck Tracking Dashboard
 */

export type JobStatus = "created" | "processing" | "completed" | "failed";

export type BodyType = "dry_van" | "reefer" | "flatbed";
export type AxleType = "standard" | "spread";
export type SmallVehicleType = "bobtail" | "box_truck" | "pickup" | "van" | "other";

export interface Job {
  id: string;
  video_path: string;
  video_name: string;
  status: JobStatus;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  truck_count?: number;
  error?: string;
}

export interface Truck {
  truck_id: string;
  unique_truck_id?: string;  // Globally unique ID across all jobs
  job_id?: string;  // Job/session identifier
  timestamp: string;
  files?: {
    frame?: string;
    crop?: string;
  };
  body_type?: BodyType;
  axle_type?: AxleType;
  small_vehicle_type?: SmallVehicleType;
  confidence?: number;
}

export interface JobDetails extends Job {
  trucks: Truck[];
  body_type_distribution?: {
    [key: string]: number;
  };
  small_vehicle_distribution?: {
    [key: string]: number;
  };
}

export interface DashboardStats {
  total_jobs: number;
  processing: number;
  completed: number;
  failed: number;
  total_trucks: number;
}
