import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';

@Component({
  selector: 'app-video-processor',
  templateUrl: './video-processor.component.html',
  styleUrls: ['./video-processor.component.css']
})
export class VideoProcessorComponent {
  selectedFile: File | null = null;
  processingState: 'idle' | 'uploading' | 'processing' | 'done' | 'error' = 'idle';
  uploadProgress: number = 0;
  resultVideoUrl: string | null = null;
  errorMessage: string = '';

  constructor(private http: HttpClient) {}

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.resultVideoUrl = null;
      this.processingState = 'idle';
    }
  }

  onUpload(): void {
    if (!this.selectedFile) return;

    this.processingState = 'uploading';
    this.uploadProgress = 0;
    
    // Create FormData
    const formData = new FormData();
    formData.append('video', this.selectedFile);

    // Make an API call to the C# Backend
    // Example endpoint: 'https://localhost:5001/api/video/process'
    this.http.post<{ resultUrl: string }>('/api/video/process', formData, {
      reportProgress: true,
      observe: 'events'
    }).pipe(
      catchError(error => {
        this.processingState = 'error';
        this.errorMessage = 'Đã có lỗi xảy ra khi xử lý video.';
        console.error(error);
        return throwError(() => error);
      })
    ).subscribe((event: any) => {
      // Simulate real progress or parse Http progress event
      if (event.type === 1) { // HttpEventType.UploadProgress
        this.uploadProgress = Math.round(100 * event.loaded / event.total);
      } else if (event.type === 4) { // HttpEventType.Response
        this.processingState = 'done';
        this.resultVideoUrl = event.body.resultUrl;
      }
    });

    // Mock Process (For UI testing without backend)
    /*
    setTimeout(() => { this.processingState = 'processing'; }, 1000);
    setTimeout(() => { 
      this.processingState = 'done'; 
      this.resultVideoUrl = 'example_output.mp4'; 
    }, 5000);
    */
  }

  downloadResult(): void {
    if (this.resultVideoUrl) {
        window.open(this.resultVideoUrl, '_blank');
    }
  }
}
