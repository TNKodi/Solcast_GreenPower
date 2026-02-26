"""
Forecast service that orchestrates the power prediction workflow.
Integrates with existing power prediction logic from Scripts/power_predicition.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

# Add the scripts directory to Python path to import existing modules
scripts_path = Path(__file__).parent.parent.parent / "Scripts" / "power_predicition"
sys.path.insert(0, str(scripts_path))

from power import power_prediction
from app.services.thingsboard_client import ThingsBoardClient
from app.services.job_manager import job_manager, JobStatus
from app.utils.logger import logger
from app.utils.config import settings


class ForecastService:
    """
    Service for executing power forecasts on assets.
    """
    
    # Required attributes for successful forecast
    REQUIRED_ATTRIBUTES = [
        "latitude",
        "longitude",
        "orientation",
        "pv_module_area",
        "inverter_power"
    ]
    
    def __init__(self):
        pass

    def _get_yesterday_range(self) -> Tuple[datetime, datetime]:
        """
        Return yesterday start/end in local timezone.
        """
        local_tz = ZoneInfo(settings.TZ_LOCAL)
        now_local = datetime.now(local_tz)
        yesterday = (now_local - timedelta(days=1)).date()

        start_dt = datetime(
            yesterday.year,
            yesterday.month,
            yesterday.day,
            0,
            0,
            0,
            tzinfo=local_tz,
        )
        end_dt = start_dt + timedelta(days=1) - timedelta(seconds=1)
        return start_dt, end_dt
    
    def validate_asset_attributes(self, attributes: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate that asset has all required attributes.
        
        Args:
            attributes: Asset attributes dictionary
            
        Returns:
            Tuple of (is_valid, missing_attributes)
        """
        missing = []
        for attr in self.REQUIRED_ATTRIBUTES:
            if attr not in attributes or attributes[attr] is None:
                missing.append(attr)
        
        return len(missing) == 0, missing

    def _normalize_bounds_for_index(self, index: pd.DatetimeIndex, start_dt, end_dt):
        """
        Normalize start/end datetimes to be compatible with a DatetimeIndex timezone.
        """
        start_ts = pd.Timestamp(start_dt)
        end_ts = pd.Timestamp(end_dt)

        if index.tz is not None:
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize(index.tz)
            else:
                start_ts = start_ts.tz_convert(index.tz)

            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize(index.tz)
            else:
                end_ts = end_ts.tz_convert(index.tz)
        else:
            if start_ts.tzinfo is not None:
                start_ts = start_ts.tz_localize(None)
            if end_ts.tzinfo is not None:
                end_ts = end_ts.tz_localize(None)

        return start_ts, end_ts
    
    async def process_single_asset(
        self, 
        asset_id: str,
        tb_client: ThingsBoardClient,
        start_dt,
        end_dt,
        already_have: bool,
        job_id: Optional[str] = None
    ) -> tuple[bool, Optional[str], Optional[List[str]]]:
        """
        Process a single asset: read attributes, run forecast, write telemetry.
        
        Args:
            asset_id: Asset ID to process
            tb_client: ThingsBoard client instance
            job_id: Optional job ID for tracking
            
        Returns:
            Tuple of (success, error_message, missing_attributes)
        """
        try:
            logger.info(f"Processing asset {asset_id}")
            
            # Step 1: Read device attributes
            logger.debug(f"Reading attributes for {asset_id}")
            attributes = await tb_client.read_device_attributes(asset_id)
            
            if not attributes:
                logger.warning(f"No attributes found for {asset_id}, skipping")
                return False, "no_attributes_found", None
            
            logger.info(f"Retrieved {len(attributes)} attributes for {asset_id}")
            
            # Step 2: Validate attributes
            is_valid, missing = self.validate_asset_attributes(attributes)
            if not is_valid:
                logger.warning(f"Asset {asset_id} missing required attributes: {missing}")
                return False, "missing_required_attributes", missing

            # Step 3: Historical data requirement when already_have is True
            if already_have:
                has_history = await tb_client.has_historical_data(asset_id, start_dt, end_dt)
                if not has_history:
                    logger.warning(
                        f"Asset {asset_id} lacks historical data in requested range; "
                        "continuing with forecast-only generation"
                    )
            
            # Step 4: Run power prediction (synchronous call to existing code)
            logger.info(f"Running power prediction for {asset_id}")
            try:
                daily_power, total_energy = power_prediction(
                    attributes,
                    start_date=start_dt.date().isoformat(),
                    end_date=end_dt.date().isoformat(),
                )
                # Filter to requested date range if possible
                if isinstance(daily_power.index, pd.DatetimeIndex):
                    start_bound, end_bound = self._normalize_bounds_for_index(daily_power.index, start_dt, end_dt)
                    daily_power = daily_power.loc[(daily_power.index >= start_bound) & (daily_power.index <= end_bound)]
                logger.info(f"Prediction completed for {asset_id}: {total_energy:.2f} kWh total")
            except Exception as e:
                logger.error(f"Power prediction failed for {asset_id}: {e}")
                return False, f"forecast_calculation_error: {str(e)}", None
            
            # Step 5: Write telemetry to ThingsBoard
            logger.info(f"Writing telemetry for {asset_id}")
            telemetry_records = self._prepare_telemetry(daily_power, asset_id, start_dt, end_dt)
            
            if telemetry_records:
                try:
                    await tb_client.write_bulk_telemetry(asset_id, telemetry_records)
                    logger.info(f"Successfully wrote {len(telemetry_records)} telemetry records for {asset_id}")
                except Exception as e:
                    logger.error(f"Telemetry write failed for {asset_id}: {e}")
                    return False, f"telemetry_write_failed: {str(e)}", None
            else:
                logger.warning(f"No telemetry records to write for {asset_id}")
            
            return True, None, None
            
        except Exception as e:
            logger.error(f"Failed to process asset {asset_id}: {e}", exc_info=True)
            return False, f"unexpected_error: {str(e)}", None
    
    def _prepare_telemetry(
        self, 
        daily_power: pd.DataFrame, 
        asset_id: str,
        start_dt,
        end_dt
    ) -> List[Dict[str, Any]]:
        """
        Prepare telemetry records from daily power DataFrame.
        
        Args:
            daily_power: DataFrame with daily energy predictions
            asset_id: Asset ID (for logging)
            
        Returns:
            List of telemetry records
        """
        telemetry_list = []
        
        try:
            # Apply date filtering just in case upstream did not filter
            if isinstance(daily_power.index, pd.DatetimeIndex):
                start_bound, end_bound = self._normalize_bounds_for_index(daily_power.index, start_dt, end_dt)
                daily_power = daily_power.loc[(daily_power.index >= start_bound) & (daily_power.index <= end_bound)]

            # Get the energy column (first column typically contains energy_kwh)
            if 'energy_kwh' in daily_power.columns:
                energy_column = 'energy_kwh'
            else:
                # Use first column
                energy_column = daily_power.columns[0]
            
            for i in range(len(daily_power)):
                timestamp = daily_power.index[i]
                energy_value = daily_power.iloc[i][energy_column]
                
                # Convert timestamp to milliseconds
                ts_ms = int(timestamp.timestamp() * 1000)
                
                telemetry_record = {
                    "ts": ts_ms,
                    "values": {
                        "solcast_daily": float(energy_value)
                    }
                }
                
                telemetry_list.append(telemetry_record)
            
            logger.debug(f"Prepared {len(telemetry_list)} telemetry records for {asset_id}")
            
        except Exception as e:
            logger.error(f"Error preparing telemetry for {asset_id}: {e}")
        
        return telemetry_list
    
    async def run_forecast_for_main_asset(
        self,
        job_id: str,
        main_asset_id: str,
        already_have: bool
    ) -> None:
        """
        Execute forecast workflow for a main asset and all its children.
        This is the long-running background task.
        
        Args:
            job_id: Job ID for tracking
            main_asset_id: Main asset ID to start from
            already_have: Whether historical data exists
        """
        try:
            start_dt, end_dt = self._get_yesterday_range()
            logger.info(
                f"Starting forecast job {job_id} for main asset {main_asset_id} | "
                f"already_have={already_have} | start={start_dt.isoformat()} end={end_dt.isoformat()}"
            )
            
            # Mark job as running
            await job_manager.mark_running(job_id)
            
            async with ThingsBoardClient() as tb_client:
                # Step 1: Get all related assets
                logger.info(f"Retrieving assets from {main_asset_id}")
                await job_manager.update_job_progress(
                    job_id,
                    progress_message="Retrieving asset hierarchy"
                )
                
                target_level = settings.DEFAULT_TARGET_LEVEL
                assets = await tb_client.get_assets_by_level(main_asset_id, target_level)
                
                logger.info(f"Found {len(assets)} assets at level {target_level}")
                await job_manager.update_job_progress(
                    job_id,
                    total_assets=len(assets),
                    assets_processed=0,
                    successful=0,
                    failed=0,
                    skipped=0,
                    progress_message=f"Processing {len(assets)} assets",
                    current_step="asset_traversal_complete"
                )
                
                if not assets:
                    logger.warning(f"No assets found for {main_asset_id}")
                    await job_manager.mark_completed(job_id)
                    return
                
                # Step 2: Process each asset
                successful = 0
                failed = 0
                skipped = 0
                
                for idx, asset_id in enumerate(assets, 1):
                    logger.info(f"Processing asset {idx}/{len(assets)}: {asset_id}")
                    
                    await job_manager.update_job_progress(
                        job_id,
                        current_step=f"processing_asset_{idx}"
                    )
                    
                    success, error_msg, missing_attrs = await self.process_single_asset(
                        asset_id, tb_client, start_dt, end_dt, already_have, job_id
                    )
                    
                    if success:
                        successful += 1
                    elif error_msg and ("missing_required_attributes" in error_msg or "historical_data_not_available" in error_msg):
                        skipped += 1
                        await job_manager.add_job_error(
                            job_id, asset_id, error_msg, missing_attrs
                        )
                    else:
                        failed += 1
                        await job_manager.add_job_error(
                            job_id, asset_id, error_msg or "unknown_error"
                        )
                    
                    # Update job progress
                    await job_manager.update_job_progress(
                        job_id,
                        assets_processed=idx,
                        successful=successful,
                        failed=failed,
                        skipped=skipped,
                        progress_message=f"Processed {idx}/{len(assets)} (success: {successful}, failed: {failed}, skipped: {skipped})"
                    )
                
                # Step 3: Mark job as completed
                logger.info(f"Forecast job {job_id} completed: {successful} successful, {failed} failed, {skipped} skipped")
                await job_manager.mark_completed(job_id)
                
        except Exception as e:
            error_msg = f"Forecast job {job_id} failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await job_manager.mark_failed(job_id, error_msg)
    
    async def process_new_asset(self, asset_id: str, already_have: bool) -> Dict[str, Any]:
        """
        Process a single new asset (synchronous execution).
        
        Args:
            asset_id: Asset ID to process
            already_have: Whether asset already has historical data
            
        Returns:
            Result dictionary with status and message
        """
        try:
            start_dt, end_dt = self._get_yesterday_range()
            logger.info(
                f"Processing new asset {asset_id} | already_have={already_have} | "
                f"start={start_dt.isoformat()} end={end_dt.isoformat()}"
            )
            
            async with ThingsBoardClient() as tb_client:
                # Check if asset exists
                exists = await tb_client.asset_exists(asset_id)
                if not exists:
                    return {
                        "status": "error",
                        "asset_id": asset_id,
                        "error": "ASSET_NOT_FOUND",
                        "message": "The provided asset_id does not exist in ThingsBoard"
                    }
                
                success, error_msg, missing_attrs = await self.process_single_asset(
                    asset_id, tb_client, start_dt, end_dt, already_have
                )
                
                if success:
                    return {
                        "status": "telemetry_written",
                        "asset_id": asset_id,
                        "message": "Successfully processed new asset"
                    }
                elif error_msg and "missing_required_attributes" in error_msg:
                    return {
                        "status": "error",
                        "asset_id": asset_id,
                        "error": "ASSET_MISSING_ATTRIBUTES",
                        "message": "Asset exists but required attributes are missing",
                        "missing_attributes": missing_attrs or []
                    }
                else:
                    return {
                        "status": "failed",
                        "asset_id": asset_id,
                        "message": error_msg or "Failed to process asset"
                    }
                    
        except Exception as e:
            logger.error(f"Error processing new asset {asset_id}: {e}", exc_info=True)
            return {
                "status": "error",
                "asset_id": asset_id,
                "message": str(e)
            }


# Global forecast service instance
forecast_service = ForecastService()
