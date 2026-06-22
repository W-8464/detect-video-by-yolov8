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
  folderName: string | null = null;
  actions: any[] = [];
  selectedActionIndices: number[] = [];
  hasChanges: boolean = false;

  // Render options
  showBox: boolean = true;
  showHandPose: boolean = true;

  constructor(private http: HttpClient) {}

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.resultVideoUrl = null;
      this.processingState = 'idle';
      this.actions = [];
      this.fileId = null;
      this.folderName = null;
      this.hasChanges = false;
    }
  }

  onUpload(): void {
    if (!this.selectedFile) return;

    this.processingState = 'uploading';
    this.uploadProgress = 0;
    this.hasChanges = false;
    
    const formData = new FormData();
    formData.append('video', this.selectedFile);

    this.http.post<{ fileId: string, folderName: string, actions: any[] }>('/api/video/process', formData, {
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
        this.folderName = event.body.folderName;
        this.actions = event.body.actions;
        this.resultVideoUrl = event.body.resultUrl; // Initial SOP video
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

    this.hasChanges = true;
    const firstIdx = this.selectedActionIndices[0];
    const lastIdx = this.selectedActionIndices[this.selectedActionIndices.length - 1];
    
    const newAction = {
      action_id: 'merged_action',
      start_frame: this.actions[firstIdx].start_frame,
      end_frame: this.actions[lastIdx].end_frame
    };

    const newActionsList = [...this.actions];
    const sortedIndices = [...this.selectedActionIndices].sort((a, b) => b - a);
    for (const idx of sortedIndices) {
      newActionsList.splice(idx, 1);
    }
    
    newActionsList.splice(firstIdx, 0, newAction);
    
    this.actions = newActionsList;
    this.selectedActionIndices = [];
  }

  renameAction(index: number, newName: string): void {
    if (newName && newName.trim() && this.actions[index].action_id !== newName.trim()) {
      this.actions[index].action_id = newName.trim();
      this.hasChanges = true;
    }
  }

  onRender(): void {
    if (!this.fileId || !this.folderName || this.actions.length === 0) return;

    this.processingState = 'rendering';
    
    // We should always render if onRender is called, because the user might have changed 
    // the render options (box/hand-pose) even if they didn't edit the actions.

    this.http.post<{ resultUrl: string }>('/api/video/render', {
      file_id: this.fileId,
      folder_name: this.folderName,
      actions: this.actions,
      show_box: this.showBox,
      show_hand_pose: this.showHandPose
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
