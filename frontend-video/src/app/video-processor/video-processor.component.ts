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
  processingState: 'idle' | 'uploading' | 'processing' | 'editing' | 'rendering' | 'done' | 'error' = 'idle';
  uploadProgress: number = 0;
  resultVideoUrl: string | null = null;
  errorMessage: string = '';
  
  // Action management
  fileId: string | null = null;
  actions: any[] = [];
  selectedActionIndices: number[] = [];

  constructor(private http: HttpClient) {}

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.resultVideoUrl = null;
      this.processingState = 'idle';
      this.actions = [];
      this.fileId = null;
    }
  }

  onUpload(): void {
    if (!this.selectedFile) return;

    this.processingState = 'uploading';
    this.uploadProgress = 0;
    
    const formData = new FormData();
    formData.append('video', this.selectedFile);

    this.http.post<{ fileId: string, actions: any[] }>('/api/video/process', formData, {
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
      if (event.type === 1) { // HttpEventType.UploadProgress
        this.uploadProgress = Math.round(100 * event.loaded / event.total);
        if (this.uploadProgress === 100) {
            this.processingState = 'processing';
        }
      } else if (event.type === 4) { // HttpEventType.Response
        this.fileId = event.body.fileId;
        this.actions = event.body.actions;
        this.processingState = 'editing';
      }
    });
  }

  toggleActionSelection(index: number): void {
    const pos = this.selectedActionIndices.indexOf(index);
    if (pos > -1) {
      this.selectedActionIndices.splice(pos, 1);
    } else {
      this.selectedActionIndices.push(index);
      this.selectedActionIndices.sort((a, b) => a - b);
    }
  }

  mergeActions(): void {
    if (this.selectedActionIndices.length < 2) return;

    const firstIdx = this.selectedActionIndices[0];
    const lastIdx = this.selectedActionIndices[this.selectedActionIndices.length - 1];
    
    // Check if they are contiguous in the current actions list
    // Actually, user said merge regardless of idle in between
    const newAction = {
      action_id: 'merged_action',
      start_frame: this.actions[firstIdx].start_frame,
      end_frame: this.actions[lastIdx].end_frame
    };

    // Remove old actions and insert new one
    const newActionsList = [...this.actions];
    // We need to remove from highest index to lowest to maintain indices
    const sortedIndices = [...this.selectedActionIndices].sort((a, b) => b - a);
    for (const idx of sortedIndices) {
      newActionsList.splice(idx, 1);
    }
    
    // Insert at the position of the first selected action
    newActionsList.splice(firstIdx, 0, newAction);
    
    this.actions = newActionsList;
    this.selectedActionIndices = [];
  }

  renameAction(index: number, newName: string): void {
    if (newName && newName.trim()) {
      this.actions[index].action_id = newName.trim();
    }
  }

  onRender(): void {
    if (!this.fileId || this.actions.length === 0) return;

    this.processingState = 'rendering';
    
    this.http.post<{ resultUrl: string }>('/api/video/render', {
      file_id: this.fileId,
      actions: this.actions
    }).pipe(
      catchError(error => {
        this.processingState = 'error';
        this.errorMessage = 'Đã có lỗi xảy ra khi render video.';
        console.error(error);
        return throwError(() => error);
      })
    ).subscribe(res => {
      this.resultVideoUrl = res.resultUrl;
      this.processingState = 'done';
    });
  }

  downloadResult(): void {
    if (this.resultVideoUrl) {
        window.open(this.resultVideoUrl, '_blank');
    }
  }
}
