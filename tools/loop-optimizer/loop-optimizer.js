/**
 * LoopOptimizer: A tool for optimizing video loops
 * Version: 0.2
 * 
 * This tool identifies optimal loop points in video clips and ensures
 * smooth transitions for repeatable playback.
 */

const fs = require('fs');
const path = require('path');
const { exec, spawn } = require('child_process');
const { v4: uuidv4 } = require('uuid');
const dotenv = require('dotenv');
const { Storage } = require('@google-cloud/storage');
const { VideoIntelligenceServiceClient } = require('@google-cloud/video-intelligence').v1;

// Load environment variables
dotenv.config();

/**
 * Main LoopOptimizer class
 */
class LoopOptimizer {
  /**
   * Constructor
   * @param {Object} options - Configuration options
   */
  constructor(options = {}) {
    // Set default configuration
    this.config = {
      similarityThreshold: options.similarityThreshold || process.env.DEFAULT_SIMILARITY_THRESHOLD || 0.9,
      crossfadeDuration: options.crossfadeDuration || process.env.DEFAULT_CROSSFADE_DURATION || 0.5,
      outputDirectory: options.outputDirectory || process.env.DEFAULT_OUTPUT_DIRECTORY || './output',
      minimumLoopLength: options.minimumLoopLength || 1.0,
      maximumLoopLength: options.maximumLoopLength || null, // null means full length
      generateThumbnails: options.generateThumbnails !== undefined ? options.generateThumbnails : true,
      optimizeFor: options.optimizeFor || 'quality', // 'quality', 'file_size', or 'processing_speed'
      tempDirectory: options.tempDirectory || './tmp',
      useGoogleCloud: options.useGoogleCloud !== undefined ? options.useGoogleCloud : true,
    };

    // Create output and temp directories if they don't exist
    this._ensureDirectoryExists(this.config.outputDirectory);
    this._ensureDirectoryExists(this.config.tempDirectory);

    // Initialize logging
    this.logDirectory = options.logDirectory || path.join(
      process.platform === 'win32' ? process.env.USERPROFILE : process.env.HOME, 
      '.loop-optimizer', 
      'logs'
    );
    this._ensureDirectoryExists(this.logDirectory);
    this.logFile = path.join(this.logDirectory, `loop-optimizer-${new Date().toISOString().split('T')[0]}.log`);

    this.log('LoopOptimizer initialized with configuration:', this.config);

    // Initialize Google Cloud clients if available
    if (this.config.useGoogleCloud) {
      try {
        this.videoIntelligenceClient = new VideoIntelligenceServiceClient();
        this.storage = new Storage();
        this.log('Google Cloud clients initialized');
      } catch (error) {
        this.log('Error initializing Google Cloud clients:', error, 'error');
        this.config.useGoogleCloud = false;
      }
    }
  }

  /**
   * Logs a message to file and console
   * @param {string} message - The main message
   * @param {any} data - Additional data to log
   * @param {string} level - Log level (info, warn, error)
   */
  log(message, data = null, level = 'info') {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] [${level.toUpperCase()}] ${message}`;
    
    // Log to console
    if (level === 'error') {
      console.error(logMessage);
      if (data) console.error(data);
    } else if (level === 'warn') {
      console.warn(logMessage);
      if (data) console.warn(data);
    } else {
      console.log(logMessage);
      if (data) console.log(data);
    }

    // Log to file
    try {
      const logEntry = `${logMessage}${data ? '\n' + JSON.stringify(data, null, 2) : ''}\n`;
      fs.appendFileSync(this.logFile, logEntry);
    } catch (error) {
      console.error(`Error writing to log file: ${error.message}`);
    }
  }

  /**
   * Ensures a directory exists, creating it if necessary
   * @param {string} directory - Directory path
   */
  _ensureDirectoryExists(directory) {
    if (!fs.existsSync(directory)) {
      fs.mkdirSync(directory, { recursive: true });
    }
  }

  /**
   * Main method to optimize a video
   * @param {string} videoPath - Path to the video file
   * @param {Object} options - Override default options
   * @returns {Promise<Object>} - Results of the optimization
   */
  async optimizeVideo(videoPath, options = {}) {
    // Merge options with defaults
    const config = { ...this.config, ...options };
    
    // Generate a unique ID for this video processing job
    const videoId = path.basename(videoPath, path.extname(videoPath));
    const jobId = uuidv4();
    
    this.log(`Starting video optimization for ${videoPath}`, { videoId, jobId });
    
    try {
      // Check if video exists and is readable
      if (!fs.existsSync(videoPath)) {
        throw new Error(`Video file not found: ${videoPath}`);
      }
      
      // Get video information
      const videoInfo = await this._getVideoInfo(videoPath);
      this.log('Video information retrieved', videoInfo);
      
      // Detect loop points
      const loopPoints = await this.detectLoopPoints(videoPath, config);
      this.log(`Detected ${loopPoints.length} potential loop points`, loopPoints);
      
      if (loopPoints.length === 0) {
        throw new Error('No suitable loop points found (Error 2001). Try reducing similarity threshold.');
      }
      
      // Create loops from the detected points
      const loops = await this._createLoops(videoPath, loopPoints, config);
      this.log(`Created ${loops.length} loops`);
      
      // Determine the recommended loop
      const recommendedLoop = this._getRecommendedLoop(loops);
      this.log('Recommended loop:', recommendedLoop);
      
      // Generate thumbnails if requested
      let thumbnailDirectory = null;
      if (config.generateThumbnails) {
        thumbnailDirectory = await this._generatePreviews(loops, videoId, config);
        this.log(`Generated thumbnails in ${thumbnailDirectory}`);
      }
      
      // Prepare result
      const result = {
        video_id: videoId,
        original_duration: videoInfo.duration,
        processing_timestamp: new Date().toISOString(),
        loops,
        recommended_loop: recommendedLoop,
        thumbnail_directory: thumbnailDirectory
      };
      
      this.log('Video optimization completed successfully', { videoId, jobId });
      return result;
    } catch (error) {
      this.log(`Error optimizing video: ${error.message}`, error.stack, 'error');
      throw error;
    }
  }

  /**
   * Get information about a video file using FFmpeg
   * @param {string} videoPath - Path to the video file
   * @returns {Promise<Object>} - Video information
   */
  _getVideoInfo(videoPath) {
    return new Promise((resolve, reject) => {
      exec(`ffprobe -v quiet -print_format json -show_format -show_streams "${videoPath}"`, (error, stdout) => {
        if (error) {
          reject(new Error(`Failed to get video information: ${error.message} (Error 1001)`));
          return;
        }
        
        try {
          const info = JSON.parse(stdout);
          
          // Extract relevant information
          const videoStream = info.streams.find(stream => stream.codec_type === 'video');
          const audioStream = info.streams.find(stream => stream.codec_type === 'audio');
          
          if (!videoStream) {
            reject(new Error('No video stream found in file (Error 1002)'));
            return;
          }
          
          // Calculate duration
          const duration = parseFloat(info.format.duration);
          
          // Get resolution
          const width = parseInt(videoStream.width);
          const height = parseInt(videoStream.height);
          
          // Get framerate
          const frameRate = eval(videoStream.r_frame_rate);
          
          // Get codec
          const videoCodec = videoStream.codec_name;
          const audioCodec = audioStream ? audioStream.codec_name : null;
          
          resolve({
            duration,
            width,
            height,
            frameRate,
            videoCodec,
            audioCodec,
            hasAudio: !!audioStream
          });
        } catch (parseError) {
          reject(new Error(`Failed to parse video information: ${parseError.message} (Error 1003)`));
        }
      });
    });
  }

  /**
   * Detect potential loop points in a video
   * @param {string} videoPath - Path to the video file
   * @param {Object} options - Configuration options
   * @returns {Promise<Array>} - Detected loop points
   */
  async detectLoopPoints(videoPath, options = {}) {
    const config = { ...this.config, ...options };
    this.log('Detecting loop points', { videoPath, config });
    
    // Get video information
    const videoInfo = await this._getVideoInfo(videoPath);
    
    // Determine loop length constraints
    const minLoopLength = config.minimumLoopLength;
    const maxLoopLength = config.maximumLoopLength || videoInfo.duration;
    
    // Use different strategies based on configuration
    if (config.useGoogleCloud && this.videoIntelligenceClient) {
      return this._detectLoopPointsWithGoogleCloud(videoPath, videoInfo, minLoopLength, maxLoopLength, config);
    } else {
      return this._detectLoopPointsWithFFmpeg(videoPath, videoInfo, minLoopLength, maxLoopLength, config);
    }
  }

  /**
   * Detect loop points using Google Cloud Video Intelligence
   * @param {string} videoPath - Path to the video file
   * @param {Object} videoInfo - Video information
   * @param {number} minLoopLength - Minimum loop length in seconds
   * @param {number} maxLoopLength - Maximum loop length in seconds
   * @param {Object} config - Configuration options
   * @returns {Promise<Array>} - Detected loop points
   */
  async _detectLoopPointsWithGoogleCloud(videoPath, videoInfo, minLoopLength, maxLoopLength, config) {
    try {
      this.log('Using Google Cloud Video Intelligence API for loop detection');
      
      // Upload the file to GCS temporarily if it's a local file
      let gcsUri;
      let tempBucketName;
      
      if (!videoPath.startsWith('gs://')) {
        tempBucketName = `loop-optimizer-temp-${Date.now()}`;
        const bucket = await this.storage.createBucket(tempBucketName);
        const fileName = path.basename(videoPath);
        
        await bucket.upload(videoPath);
        gcsUri = `gs://${tempBucketName}/${fileName}`;
        this.log(`Uploaded video to ${gcsUri}`);
      } else {
        gcsUri = videoPath;
      }
      
      // Configure the request
      const request = {
        inputUri: gcsUri,
        features: ['SHOT_CHANGE_DETECTION']
      };
      
      // Make the request
      const [operation] = await this.videoIntelligenceClient.annotateVideo(request);
      this.log('Video Intelligence API request submitted, waiting for results...');
      
      // Wait for operation to complete
      const [response] = await operation.promise();
      
      // Process shot change annotations
      const shotChanges = response.annotationResults[0].shotAnnotations || [];
      this.log(`Detected ${shotChanges.length} shot changes`);
      
      // Clean up temporary GCS bucket if created
      if (tempBucketName) {
        const bucket = this.storage.bucket(tempBucketName);
        await bucket.deleteFiles({ force: true });
        await bucket.delete();
        this.log(`Cleaned up temporary GCS bucket ${tempBucketName}`);
      }
      
      // If no shot changes are detected, use FFmpeg method as fallback
      if (shotChanges.length === 0) {
        this.log('No shot changes detected by Video Intelligence API, falling back to FFmpeg');
        return this._detectLoopPointsWithFFmpeg(videoPath, videoInfo, minLoopLength, maxLoopLength, config);
      }
      
      // Convert shot changes to potential loop points
      const loopPoints = [];
      
      for (let i = 0; i < shotChanges.length; i++) {
        const startShot = shotChanges[i];
        
        // Find shots that would create loops within our length constraints
        for (let j = i + 1; j < shotChanges.length; j++) {
          const endShot = shotChanges[j];
          const startTime = startShot.startTimeOffset.seconds + startShot.startTimeOffset.nanos / 1e9;
          const endTime = endShot.startTimeOffset.seconds + endShot.startTimeOffset.nanos / 1e9;
          const duration = endTime - startTime;
          
          if (duration >= minLoopLength && duration <= maxLoopLength) {
            // We'll analyze frame similarity later when creating the actual loops
            loopPoints.push({
              start_time: startTime,
              end_time: endTime,
              duration: duration,
              similarity_score: null  // To be calculated later
            });
          }
        }
      }
      
      // If still no loop points, use frame analysis method
      if (loopPoints.length === 0) {
        this.log('No suitable loop points from shot detection, falling back to FFmpeg');
        return this._detectLoopPointsWithFFmpeg(videoPath, videoInfo, minLoopLength, maxLoopLength, config);
      }
      
      return loopPoints;
    } catch (error) {
      this.log('Error using Google Cloud Video Intelligence:', error, 'error');
      this.log('Falling back to FFmpeg for loop detection');
      return this._detectLoopPointsWithFFmpeg(videoPath, videoInfo, minLoopLength, maxLoopLength, config);
    }
  }

  /**
   * Detect loop points using FFmpeg
   * @param {string} videoPath - Path to the video file
   * @param {Object} videoInfo - Video information
   * @param {number} minLoopLength - Minimum loop length in seconds
   * @param {number} maxLoopLength - Maximum loop length in seconds
   * @param {Object} config - Configuration options
   * @returns {Promise<Array>} - Detected loop points
   */
  async _detectLoopPointsWithFFmpeg(videoPath, videoInfo, minLoopLength, maxLoopLength, config) {
    this.log('Using FFmpeg for loop detection');
    
    // Calculate frame extraction interval based on video duration and optimization setting
    let frameInterval;
    if (config.optimizeFor === 'processing_speed') {
      // Extract fewer frames for faster processing
      frameInterval = Math.max(1, Math.ceil(videoInfo.frameRate / 4));
    } else if (config.optimizeFor === 'file_size') {
      // Medium precision
      frameInterval = Math.max(1, Math.ceil(videoInfo.frameRate / 8));
    } else {
      // High precision for quality
      frameInterval = Math.max(1, Math.ceil(videoInfo.frameRate / 15));
    }
    
    // Extract frames for analysis
    const framesDir = path.join(this.config.tempDirectory, `frames_${Date.now()}`);
    this._ensureDirectoryExists(framesDir);
    
    this.log(`Extracting frames at interval ${frameInterval}`, { framesDir });
    
    await new Promise((resolve, reject) => {
      exec(
        `ffmpeg -i "${videoPath}" -vf "select=not(mod(n\\,${frameInterval}))" -vsync vfr "${framesDir}/frame_%04d.jpg"`,
        (error) => {
          if (error) {
            reject(new Error(`Frame extraction failed: ${error.message} (Error 1004)`));
          } else {
            resolve();
          }
        }
      );
    });
    
    // Get list of extracted frames
    const frameFiles = fs.readdirSync(framesDir)
      .filter(file => file.startsWith('frame_') && file.endsWith('.jpg'))
      .sort();
    
    this.log(`Extracted ${frameFiles.length} frames`);
    
    if (frameFiles.length < 2) {
      throw new Error('Not enough frames extracted for analysis (Error 1005)');
    }
    
    // Calculate frame time mapping
    const frameTimeMap = frameFiles.map((file, index) => {
      const frameNumber = parseInt(file.replace('frame_', '').replace('.jpg', ''));
      const approximateTime = (frameNumber * frameInterval) / videoInfo.frameRate;
      return { 
        file: path.join(framesDir, file), 
        time: approximateTime,
        index
      };
    });
    
    // Find potential loop points by comparing frames
    const loopPoints = [];
    const similarityThreshold = config.similarityThreshold;
    
    // Determine comparison range based on min/max loop length
    const minFrameDistance = Math.floor(minLoopLength * videoInfo.frameRate / frameInterval);
    const maxFrameDistance = Math.ceil(maxLoopLength * videoInfo.frameRate / frameInterval);
    
    this.log('Analyzing frames for similarity', { 
      minFrameDistance, 
      maxFrameDistance, 
      similarityThreshold 
    });
    
    // Compare frames to find potential loops
    for (let i = 0; i < frameTimeMap.length - minFrameDistance; i++) {
      const startFrame = frameTimeMap[i];
      
      // Determine the maximum frame to compare with
      const maxCompareIdx = Math.min(
        i + maxFrameDistance,
        frameTimeMap.length - 1
      );
      
      // Start comparing from the minimum distance
      for (let j = i + minFrameDistance; j <= maxCompareIdx; j++) {
        const endFrame = frameTimeMap[j];
        
        // Calculate the similarity between the start and end frames
        const similarity = await this._calculateFrameSimilarity(startFrame.file, endFrame.file);
        
        if (similarity >= similarityThreshold) {
          loopPoints.push({
            start_time: startFrame.time,
            end_time: endFrame.time,
            duration: endFrame.time - startFrame.time,
            similarity_score: similarity
          });
          
          // If we're looking for speed, we can skip ahead once we find a match
          if (config.optimizeFor === 'processing_speed' && similarity > similarityThreshold + 0.05) {
            i = i + Math.floor(minFrameDistance / 2);
            break;
          }
        }
      }
    }
    
    // Clean up extracted frames
    this._cleanupDirectory(framesDir);
    
    this.log(`Found ${loopPoints.length} potential loop points`, { 
      firstFew: loopPoints.slice(0, 3) 
    });
    
    return loopPoints;
  }

  /**
   * Calculate similarity between two frames
   * @param {string} frame1Path - Path to first frame
   * @param {string} frame2Path - Path to second frame
   * @returns {Promise<number>} - Similarity score (0-1)
   */
  async _calculateFrameSimilarity(frame1Path, frame2Path) {
    // Use ImageMagick's compare command to calculate similarity
    return new Promise((resolve, reject) => {
      exec(
        `magick compare -metric RMSE "${frame1Path}" "${frame2Path}" null:`,
        { encoding: 'utf8' },
        (error, stdout, stderr) => {
          // ImageMagick outputs to stderr, not stdout
          try {
            // Extract the similarity from the output (format: "X.X (Y.Y)")
            const match = stderr.match(/(\d+\.?\d*)/);
            if (match) {
              // Convert RMSE to a similarity score (0-1)
              const rmse = parseFloat(match[1]);
              // Normalize RMSE to a similarity score (lower RMSE = higher similarity)
              const similarity = Math.max(0, 1 - (rmse / 30));
              resolve(similarity);
            } else {
              // If we can't parse the output, assume low similarity
              resolve(0);
            }
          } catch (e) {
            this.log('Error calculating frame similarity:', e, 'error');
            resolve(0);
          }
        }
      );
    });
  }

  /**
   * Create loops from the detected loop points
   * @param {string} videoPath - Path to the video file
   * @param {Array} loopPoints - Detected loop points
   * @param {Object} config - Configuration options
   * @returns {Promise<Array>} - Created loops
   */
  async _createLoops(videoPath, loopPoints, config) {
    this.log(`Creating loops from ${loopPoints.length} detected points`);
    
    const loops = [];
    const videoBaseName = path.basename(videoPath, path.extname(videoPath));
    const outputFormat = config.outputFormat || path.extname(videoPath).substring(1) || 'mp4';
    
    // Process each loop point
    for (let i = 0; i < loopPoints.length; i++) {
      const loopPoint = loopPoints[i];
      const loopFileName = `${videoBaseName}_loop_${i + 1}.${outputFormat}`;
      const outputPath = path.join(config.outputDirectory, loopFileName);
      
      try {
        // Create the loop with a crossfade
        await this._createLoopWithCrossfade(
          videoPath,
          outputPath,
          loopPoint.start_time,
          loopPoint.end_time,
          config.crossfadeDuration
        );
        
        // Evaluate the loop smoothness
        const smoothnessRating = await this._evaluateLoopSmoothness(outputPath);
        
        loops.push({
          start_time: loopPoint.start_time,
          end_time: loopPoint.end_time,
          duration: loopPoint.duration,
          similarity_score: loopPoint.similarity_score,
          smoothness_rating: smoothnessRating,
          file_path: outputPath
        });
        
        this.log(`Created loop ${i + 1}/${loopPoints.length}`, {
          file_path: outputPath,
          duration: loopPoint.duration,
          similarity_score: loopPoint.similarity_score,
          smoothness_rating: smoothnessRating
        });
      } catch (error) {
        this.log(`Failed to create loop ${i + 1}/${loopPoints.length}:`, error, 'error');
      }
    }
    
    return loops;
  }

  /**
   * Create a loop with crossfade transition
   * @param {string} inputPath - Path to input video
   * @param {string} outputPath - Path for output loop
   * @param {number} startTime - Loop start time in seconds
   * @param {number} endTime - Loop end time in seconds
   * @param {number} crossfadeDuration - Crossfade duration in seconds
   * @returns {Promise<void>}
   */
  _createLoopWithCrossfade(inputPath, outputPath, startTime, endTime, crossfadeDuration) {
    return new Promise((resolve, reject) => {
      // Calculate actual crossfade times
      const fadeDuration = Math.min(crossfadeDuration, (endTime - startTime) / 4);
      
      // Create complex FFmpeg filter for crossfade looping
      const complexFilter = [
        // Extract two segments: main segment and the overlap for crossfade
        `[0:v]trim=start=${startTime}:end=${endTime - fadeDuration},setpts=PTS-STARTPTS[main]`,
        `[0:v]trim=start=${endTime - fadeDuration}:end=${endTime},setpts=PTS-STARTPTS[fadeout]`,
        `[0:v]trim=start=${startTime}:end=${startTime + fadeDuration},setpts=PTS-STARTPTS[fadein]`,
        
        // Create crossfade between end and beginning
        `[fadeout][fadein]blend=all_expr='A*(1-T)+B*T':duration=${fadeDuration}[crossfade]`,
        
        // Concatenate main segment with crossfade
        `[main][crossfade]concat=n=2:v=1:a=0[outv]`
      ].join(';');
      
      // Handle audio if present
      const audioFilter = [
        `[0:a]atrim=start=${startTime}:end=${endTime - fadeDuration},asetpts=PTS-STARTPTS[mainsnd]`,
        `[0:a]atrim=start=${endTime - fadeDuration}:end=${endTime},asetpts=PTS-STARTPTS[fadeoutsnd]`,
        `[0:a]atrim=start=${startTime}:end=${startTime + fadeDuration},asetpts=PTS-STARTPTS[fadeinsnd]`,
        `[fadeoutsnd][fadeinsnd]acrossfade=d=${fadeDuration}[crossfadesnd]`,
        `[mainsnd][crossfadesnd]concat=n=2:v=0:a=1[outa]`
      ].join(';');
      
      // Combine filters based on whether audio is present
      const audioCheck = `ffprobe -v error -select_streams a -show_entries stream=codec_type -of csv=p=0 "${inputPath}"`;
      
      exec(audioCheck, (error, stdout) => {
        const hasAudio = !error && stdout.trim() === 'audio';
        
        let ffmpegCmd;
        if (hasAudio) {
          ffmpegCmd = [
            `ffmpeg -y -i "${inputPath}"`,
            `-filter_complex "${complexFilter};${audioFilter}"`,
            `-map "[outv]" -map "[outa]"`,
            `-c:v libx264 -preset medium -crf 22`,
            `-c:a aac -b:a 128k`,
            `"${outputPath}"`
          ].join(' ');
        } else {
          ffmpegCmd = [
            `ffmpeg -y -i "${inputPath}"`,
            `-filter_complex "${complexFilter}"`,
            `-map "[outv]"`,
            `-c:v libx264 -preset medium -crf 22`,
            `"${outputPath}"`
          ].join(' ');
        }
        
        this.log('Executing FFmpeg command', { command: ffmpegCmd });
        
        exec(ffmpegCmd, (error) => {
          if (error) {
            this.log('FFmpeg error:', error, 'error');
            reject(new Error(`Failed to create loop: ${error.message}`));
          } else {
            resolve();
          }
        });
      });
    });
  }

  /**
   * Evaluate the smoothness of a loop
   * @param {string} loopPath - Path to the loop video
   * @returns {Promise<number>} - Smoothness rating (0-1)
   */
  async _evaluateLoopSmoothness(loopPath) {
    try {
      // Extract frames from the loop point (beginning and end)
      const tempDir = path.join(this.config.tempDirectory, `loop_eval_${Date.now()}`);
      this._ensureDirectoryExists(tempDir);
      
      // Get video info to determine frame rate
      const videoInfo = await this._getVideoInfo(loopPath);
      const framesToExtract = Math.ceil(videoInfo.frameRate * 0.5); // Half a second worth of frames
      
      // Extract frames from the beginning and end of the video
      await new Promise((resolve, reject) => {
        exec(
          `ffmpeg -i "${loopPath}" -vf "select=gte(n\\,0)*lte(n\\,${framesToExtract})" -vsync 0 "${tempDir}/start_%04d.jpg"`,
          (error) => {
            if (error) reject(error);
            else resolve();
          }
        );
      });
      
      await new Promise((resolve, reject) => {
        const lastFrameCommand = `ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=noprint_wrappers=1:nokey=1 "${loopPath}"`;
        exec(lastFrameCommand, (error, stdout) => {
          if (error) {
            reject(error);
            return;
          }
          
          const totalFrames = parseInt(stdout.trim());
          const startFrame = Math.max(0, totalFrames - framesToExtract);
          
          exec(
            `ffmpeg -i "${loopPath}" -vf "select=gte(n\\,${startFrame})" -vsync 0 "${tempDir}/end_%04d.jpg"`,
            (error) => {
              if (error) reject(error);
              else resolve();
            }
          );
        });
      });
      
      // Get the extracted frames
      const startFrames = fs.readdirSync(tempDir)
        .filter(file => file.startsWith('start_'))
        .sort();
      
      const endFrames = fs.readdirSync(tempDir)
        .filter(file => file.startsWith('end_'))
        .sort();
      
      // Calculate similarity between the last end frames and first start frames
      let totalSimilarity = 0;
      const comparisonCount = Math.min(3, startFrames.length, endFrames.length);
      
      for (let i = 0; i < comparisonCount; i++) {
        const endFrame = path.join(tempDir, endFrames[endFrames.length - comparisonCount + i]);
        const startFrame = path.join(tempDir, startFrames[i]);
        
        const similarity = await this._calculateFrameSimilarity(endFrame, startFrame);
        totalSimilarity += similarity;
      }
      
      // Clean up
      this._cleanupDirectory(tempDir);
      
      // Return average similarity
      return comparisonCount > 0 ? totalSimilarity / comparisonCount : 0;
    } catch (error) {
      this.log('Error evaluating loop smoothness:', error, 'error');
      return 0;
    }
  }

  /**
   * Clean up a directory and its contents
   * @param {string} directory - Directory to clean
   */
  _cleanupDirectory(directory) {
    try {
      if (fs.existsSync(directory)) {
        const files = fs.readdirSync(directory);
        for (const file of files) {
          fs.unlinkSync(path.join(directory, file));
        }
        fs.rmdirSync(directory);
      }
    } catch (error) {
      this.log(`Error cleaning up directory ${directory}:`, error, 'warn');
    }
  }

  /**
   * Get the recommended loop from a list of loops
   * @param {Array} loops - List of loops
   * @returns {number} - Index of recommended loop
   */
  _getRecommendedLoop(loops) {
    if (loops.length === 0) return -1;
    if (loops.length === 1) return 0;
    
    // Weight factors for different metrics
    const weights = {
      duration: 0.3,        // Longer loops are generally better
      similarity: 0.4,      // Higher similarity is better
      smoothness: 0.3       // Smoother transitions are better
    };
    
    // Normalize values
    const maxDuration = Math.max(...loops.map(loop => loop.duration));
    
    // Calculate scores
    const scores = loops.map(loop => {
      const durationScore = loop.duration / maxDuration;
      const similarityScore = loop.similarity_score || 0;
      const smoothnessScore = loop.smoothness_rating || 0;
      
      return (
        weights.duration * durationScore +
        weights.similarity * similarityScore +
        weights.smoothness * smoothnessScore
      );
    });
    
    // Find the index of the highest score
    return scores.indexOf(Math.max(...scores));
  }

  /**
   * Generate preview thumbnails for loops
   * @param {Array} loops - List of loops
   * @param {string} videoId - Video identifier
   * @param {Object} config - Configuration options
   * @returns {Promise<string>} - Path to thumbnail directory
   */
  async _generatePreviews(loops, videoId, config) {
    const thumbnailDir = path.join(config.outputDirectory, `${videoId}_thumbnails`);
    this._ensureDirectoryExists(thumbnailDir);
    
    this.log(`Generating previews in ${thumbnailDir}`);
    
    for (let i = 0; i < loops.length; i++) {
      const loop = loops[i];
      const outputPath = path.join(thumbnailDir, `loop_${i + 1}.jpg`);
      
      try {
        // Extract a frame from the middle of the loop
        const middleTime = loop.start_time + (loop.duration / 2);
        
        await new Promise((resolve, reject) => {
          exec(
            `ffmpeg -ss ${middleTime} -i "${loop.file_path}" -vframes 1 -q:v 2 "${outputPath}"`,
            (error) => {
              if (error) {
                this.log(`Failed to generate thumbnail for loop ${i + 1}:`, error, 'warn');
                reject(error);
              } else {
                resolve();
              }
            }
          );
        });
      } catch (error) {
        this.log(`Error generating thumbnail for loop ${i + 1}:`, error, 'warn');
      }
    }
    
    return thumbnailDir;
  }
}

module.exports = LoopOptimizer;

// Command-line interface
if (require.main === module) {
  const yargs = require('yargs/yargs');
  const { hideBin } = require('yargs/helpers');
  
  yargs(hideBin(process.argv))
    .command(
      'optimize',
      'Optimize a video for looping',
      (yargs) => {
        return yargs
          .option('video', {
            alias: 'v',
            type: 'string',
            description: 'Path to the video file',
            demandOption: true
          })
          .option('output', {
            alias: 'o',
            type: 'string',
            description: 'Output directory'
          })
          .option('similarity', {
            alias: 's',
            type: 'number',
            description: 'Similarity threshold (0.0-1.0)'
          })
          .option('crossfade', {
            alias: 'c',
            type: 'number',
            description: 'Crossfade duration in seconds'
          })
          .option('min-length', {
            type: 'number',
            description: 'Minimum loop length in seconds'
          })
          .option('max-length', {
            type: 'number',
            description: 'Maximum loop length in seconds'
          })
          .option('no-thumbnails', {
            type: 'boolean',
            description: 'Disable thumbnail generation'
          })
          .option('optimize-for', {
            choices: ['quality', 'file_size', 'processing_speed'],
            description: 'Optimization priority'
          });
      },
      async (argv) => {
        try {
          // Configure options
          const options = {};
          if (argv.output) options.outputDirectory = argv.output;
          if (argv.similarity) options.similarityThreshold = argv.similarity;
          if (argv.crossfade) options.crossfadeDuration = argv.crossfade;
          if (argv.minLength) options.minimumLoopLength = argv.minLength;
          if (argv.maxLength) options.maximumLoopLength = argv.maxLength;
          if (argv.noThumbnails) options.generateThumbnails = false;
          if (argv.optimizeFor) options.optimizeFor = argv.optimizeFor;
          
          // Initialize and run the optimizer
          const optimizer = new LoopOptimizer(options);
          const results = await optimizer.optimizeVideo(argv.video);
          
          // Print results
          console.log('\nLoop Optimization Results:');
          console.log('=========================');
          console.log(`Video: ${argv.video}`);
          console.log(`Created ${results.loops.length} loops`);
          
          if (results.recommended_loop >= 0) {
            const recommended = results.loops[results.recommended_loop];
            console.log('\nRecommended Loop:');
            console.log(`- Duration: ${recommended.duration.toFixed(2)} seconds`);
            console.log(`- Similarity Score: ${(recommended.similarity_score * 100).toFixed(1)}%`);
            console.log(`- Smoothness Rating: ${(recommended.smoothness_rating * 100).toFixed(1)}%`);
            console.log(`- File: ${recommended.file_path}`);
          }
          
          console.log('\nAll loops saved to:', options.outputDirectory || optimizer.config.outputDirectory);
          if (results.thumbnail_directory) {
            console.log('Thumbnails saved to:', results.thumbnail_directory);
          }
        } catch (error) {
          console.error('Error:', error.message);
          process.exit(1);
        }
      }
    )
    .demandCommand(1, 'You need to specify a command')
    .help()
    .argv;
}