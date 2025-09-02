"""
AWS S3 configuration và client management.
Upload/download images to/from S3.
"""

import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from typing import Optional, Dict, Any
import logging
import os
from io import BytesIO
from PIL import Image

from .settings import settings

logger = logging.getLogger(__name__)

class S3Manager:
    """AWS S3 client manager"""
    
    def __init__(self):
        self.s3_client = None
        self.bucket_name = settings.S3_BUCKET_NAME
        self.image_prefix = settings.S3_IMAGE_PREFIX
        self._initialize_s3()
    
    def _initialize_s3(self):
        """Initialize S3 client"""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            
            # Test connection bằng cách list buckets
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 client initialized successfully for bucket: {self.bucket_name}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"S3 bucket {self.bucket_name} not found")
            else:
                logger.error(f"Failed to initialize S3 client: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing S3: {e}")
            raise
    
    def upload_image(self, local_file_path: str, s3_key: str = None) -> Dict[str, Any]:
        """
        Upload image to S3
        
        Args:
            local_file_path: Đường dẫn file local
            s3_key: S3 key (optional, sẽ auto-generate nếu không có)
        
        Returns:
            Dict chứa s3_key, s3_url, file_size, etc.
        """
        try:
            if not os.path.exists(local_file_path):
                raise FileNotFoundError(f"File not found: {local_file_path}")
            
            # Generate S3 key nếu không được provide
            if not s3_key:
                filename = os.path.basename(local_file_path)
                s3_key = f"{self.image_prefix}{filename}"
            
            # Get file info
            file_size = os.path.getsize(local_file_path)
            
            # Get image dimensions
            width, height, file_format = None, None, None
            try:
                with Image.open(local_file_path) as img:
                    width, height = img.size
                    file_format = img.format.lower()
            except Exception as e:
                logger.warning(f"Could not get image dimensions: {e}")
            
            # Upload to S3
            with open(local_file_path, 'rb') as file:
                self.s3_client.upload_fileobj(
                    file, 
                    self.bucket_name, 
                    s3_key,
                    ExtraArgs={
                        'ContentType': self._get_content_type(local_file_path),
                        'Metadata': {
                            'original_filename': os.path.basename(local_file_path),
                            'upload_source': 'admission_chatbot'
                        }
                    }
                )
            
            # Generate public URL
            s3_url = f"https://{self.bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
            
            logger.info(f"Successfully uploaded {local_file_path} to S3: {s3_key}")
            
            return {
                's3_key': s3_key,
                's3_url': s3_url,
                'file_size': file_size,
                'width': width,
                'height': height,
                'file_format': file_format
            }
            
        except Exception as e:
            logger.error(f"Failed to upload {local_file_path} to S3: {e}")
            raise
    
    def download_image(self, s3_key: str, local_file_path: str) -> bool:
        """
        Download image from S3
        
        Args:
            s3_key: S3 key của file
            local_file_path: Đường dẫn để save file
        
        Returns:
            True nếu thành công
        """
        try:
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            self.s3_client.download_file(
                self.bucket_name, 
                s3_key, 
                local_file_path
            )
            
            logger.info(f"Successfully downloaded {s3_key} to {local_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            return False
    
    def delete_image(self, s3_key: str) -> bool:
        """
        Delete image from S3
        
        Args:
            s3_key: S3 key của file cần delete
        
        Returns:
            True nếu thành công
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            logger.info(f"Successfully deleted {s3_key} from S3")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete {s3_key}: {e}")
            return False
    
    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate presigned URL cho private objects
        
        Args:
            s3_key: S3 key
            expiration: URL expiry time in seconds (default 1 hour)
        
        Returns:
            Presigned URL hoặc None nếu failed
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {s3_key}: {e}")
            return None
    
    def list_images(self, prefix: str = None) -> list:
        """
        List tất cả images trong bucket
        
        Args:
            prefix: Filter by prefix (optional)
        
        Returns:
            List of image keys
        """
        try:
            prefix = prefix or self.image_prefix
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return []
            
            return [obj['Key'] for obj in response['Contents']]
            
        except Exception as e:
            logger.error(f"Failed to list images: {e}")
            return []
    
    def _get_content_type(self, file_path: str) -> str:
        """Get content type based on file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        content_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.gif': 'image/gif'
        }
        return content_types.get(ext, 'application/octet-stream')

# Singleton instance
s3_manager = S3Manager()

# Convenience functions
def upload_image_to_s3(local_path: str, s3_key: str = None) -> Dict[str, Any]:
    """Upload image to S3"""
    return s3_manager.upload_image(local_path, s3_key)

def download_image_from_s3(s3_key: str, local_path: str) -> bool:
    """Download image from S3"""
    return s3_manager.download_image(s3_key, local_path)