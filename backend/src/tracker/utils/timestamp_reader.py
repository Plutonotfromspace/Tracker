"""
Video timestamp extraction using OCR.

Extracts timestamps from video frame overlays for accurate time tracking.
"""

import re
from datetime import datetime
from typing import Optional

import numpy as np

from src.tracker.utils.logging_config import get_logger

# Try to import easyocr, but make it optional
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

logger = get_logger(__name__)


class VideoTimestampReader:
    """
    Extracts timestamp from video frame overlay using OCR.
    
    This class uses EasyOCR to read timestamp text burned into video frames,
    typically found in security camera footage.
    
    Attributes:
        reader: EasyOCR reader instance (if available).
        last_timestamp: Cache of the last successfully extracted timestamp.
        last_frame_num: Frame number of the last cached timestamp.
    
    Example:
        >>> reader = VideoTimestampReader()
        >>> timestamp = reader.extract_timestamp(frame, frame_num=100)
        >>> if timestamp:
        ...     print(f"Video time: {timestamp}")
    """
    
    # Region of interest settings for timestamp location
    ROI_X_START_RATIO = 0.55  # Start from 55% of width
    ROI_Y_END = 80  # Top 80 pixels
    
    def __init__(self, use_gpu: bool = True) -> None:
        """
        Initialize the timestamp reader.
        
        Args:
            use_gpu: Whether to use GPU acceleration for OCR.
        """
        self.reader: Optional["easyocr.Reader"] = None
        self.last_timestamp: Optional[datetime] = None
        self.last_frame_num: int = -1
        
        if EASYOCR_AVAILABLE:
            logger.info("Initializing EasyOCR for timestamp extraction...")
            try:
                self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.reader = None
        else:
            logger.warning(
                "EasyOCR not available. Install with: pip install easyocr"
            )
    
    def extract_timestamp(
        self, 
        frame: np.ndarray, 
        frame_num: int = 0
    ) -> Optional[datetime]:
        """
        Extract timestamp from the top-right corner of the frame.
        
        Expected format: "MM-DD-YYYY HH:MM:SS AM/PM Day"
        Example: "10-16-2025 03:23:21 PM Thu"
        
        Args:
            frame: Video frame as numpy array (BGR format).
            frame_num: Current frame number for caching.
        
        Returns:
            Extracted datetime if successful, last cached timestamp otherwise.
        """
        if self.reader is None:
            return None
        
        h, w = frame.shape[:2]
        
        # Crop top-right region where timestamp is located
        roi_x1 = int(w * self.ROI_X_START_RATIO)
        roi = frame[0:self.ROI_Y_END, roi_x1:w]
        
        try:
            # Run OCR on the region
            results = self.reader.readtext(roi, detail=0, paragraph=True)
            
            if not results:
                return self.last_timestamp
            
            # Join all text found
            text = ' '.join(results)
            
            # Try to parse the timestamp
            timestamp = self._parse_timestamp(text)
            
            if timestamp:
                self.last_timestamp = timestamp
                self.last_frame_num = frame_num
                return timestamp
            
            return self.last_timestamp
            
        except Exception as e:
            logger.debug(f"OCR error: {e}")
            return self.last_timestamp
    
    def _parse_timestamp(self, text: str) -> Optional[datetime]:
        """
        Parse timestamp from OCR text.
        
        Args:
            text: OCR extracted text.
        
        Returns:
            Parsed datetime or None if parsing fails.
        """
        # Pattern: MM-DD-YYYY HH:MM:SS AM/PM (colons or periods)
        pattern_full = (
            r'(\d{1,2})-(\d{1,2})-(\d{4})\s+'
            r'(\d{1,2})[:\.](\d{2})[:\.](\d{2})\s*(AM|PM|am|pm)'
        )
        match = re.search(pattern_full, text)
        
        if match:
            return self._build_datetime(
                month=int(match.group(1)),
                day=int(match.group(2)),
                year=int(match.group(3)),
                hour=int(match.group(4)),
                minute=int(match.group(5)),
                second=int(match.group(6)),
                ampm=match.group(7).upper()
            )
        
        # Try simpler pattern without seconds
        pattern_short = (
            r'(\d{1,2})-(\d{1,2})-(\d{4})\s+'
            r'(\d{1,2})[:\.](\d{2})\s*(AM|PM|am|pm)'
        )
        match_short = re.search(pattern_short, text)
        
        if match_short:
            return self._build_datetime(
                month=int(match_short.group(1)),
                day=int(match_short.group(2)),
                year=int(match_short.group(3)),
                hour=int(match_short.group(4)),
                minute=int(match_short.group(5)),
                second=0,
                ampm=match_short.group(6).upper()
            )
        
        return None
    
    @staticmethod
    def _build_datetime(
        month: int,
        day: int,
        year: int,
        hour: int,
        minute: int,
        second: int,
        ampm: str
    ) -> datetime:
        """
        Build datetime from parsed components with AM/PM conversion.
        
        Args:
            month: Month (1-12).
            day: Day of month.
            year: Year (4 digits).
            hour: Hour (1-12).
            minute: Minute (0-59).
            second: Second (0-59).
            ampm: "AM" or "PM".
        
        Returns:
            Constructed datetime object.
        """
        # Convert to 24-hour format
        if ampm == 'PM' and hour != 12:
            hour += 12
        elif ampm == 'AM' and hour == 12:
            hour = 0
        
        return datetime(year, month, day, hour, minute, second)
    
    @property
    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self.reader is not None
