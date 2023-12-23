#define _CRT_SECURE_NO_WARNINGS // for stb

// Settings
#define DETERMINISTIC() true
static const int c_pointImageSize = 256;

static const int c_pointImageGaussSize = 512;
static const float c_pointImageGaussBlobSigma = 1.5f;

#include <random>
#include <vector>
#include <direct.h>
#include <stdio.h>
#include <chrono>

#include "squarecdf.h"
#include "NumericalCDF.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

static const int c_sampleMultiplier = 10; // 1 for debugging and iteration. 10 for quality

#define MULTITHREADED() true

struct MultiClassPoint
{
	float2 p = float2{ 0.0f, 0.0f };
	int classIndex = 0;
};

std::mt19937 GetRNG(int index)
{
	#if DETERMINISTIC()
	std::mt19937 ret(index);
	#else
	std::random_device rd;
	std::mt19937 ret(rd());
	#endif
	return ret;
}

// Circle is centered at (0,0) with radius 0.5
float UnitCircleCDF(float x)
{
	static const float c_circleArea = c_pi / 4.0f;

	// Calculate the area of the circle in [0,x]
	float ret = x * std::sqrt(0.25f - x * x) + std::asin(2.0f * x) / 4.0f;

	// Note: this is the PDF
	//float ret = 2.0f * std::sqrt(0.25f - x * x);

	// Divide by the total area to make this a PDF (integrate to 1.0)
	return ret / c_circleArea;
}

template <int NumChannels>
void PlotGaussian(std::vector<unsigned char>& image, int width, int height, int x, int y, float sigma, unsigned char color[NumChannels])
{
	int kernelRadius = int(std::sqrt(-2.0f * sigma * sigma * std::log(0.005f)));

	int sx = Clamp(x - kernelRadius, 0, width - 1);
	int ex = Clamp(x + kernelRadius, 0, height - 1);
	int sy = Clamp(y - kernelRadius, 0, width - 1);
	int ey = Clamp(y + kernelRadius, 0, height - 1);

	for (int iy = sy; iy <= ey; ++iy)
	{
		unsigned char* pixel = &image[(iy * width + sx) * NumChannels];

		int ky = std::abs(iy - y);
		float kernelY = std::exp(-float(ky * ky) / (2.0f * sigma * sigma));

		for (int ix = sx; ix <= ex; ++ix)
		{
			int kx = std::abs(ix - x);
			float kernelX = std::exp(-float(kx * kx) / (2.0f * sigma * sigma));

			float kernel = kernelX * kernelY;

			for (int i = 0; i < NumChannels; ++i)
			{
				unsigned char oldColor = *pixel;
				unsigned char newColor = (unsigned char)Lerp(float(oldColor), float(color[i]), kernel);
				*pixel = newColor;
				pixel++;
			}
		}
	}
}

void SavePointSet(const std::vector<MultiClassPoint>& points, const char* baseFileName, int index, int total)
{
	// Write out points in text
	{
		char fileName[1024];
		sprintf_s(fileName, "%s_%i_%i.txt", baseFileName, index, total);
		FILE* file = nullptr;
		fopen_s(&file, fileName, "wb");

		fprintf(file, "float points[][3] =\n{\n");

		for (size_t index = 0; index < points.size(); ++index)
			fprintf(file, "    { %ff, %ff, %ff },\n", points[index].p.x, points[index].p.y, (float)points[index].classIndex);

		fprintf(file, "};\n");

		fclose(file);
	}

	// see which classes have points, as they might not all have points
	bool classHasPoints[3] = { false, false, false };
	for (size_t index = 0; index < points.size(); ++index)
		classHasPoints[points[index].classIndex] = true;

	// Draw an image of each class
	for (int i = 1; i < 8; ++i)
	{
		bool showImage = true;
		for (int classIndex = 0; classIndex < 3; ++classIndex)
		{
			int classMask = 1 << classIndex;
			if ((i & classMask) != 0 && !classHasPoints[classIndex])
				showImage = false;
		}

		if (!showImage)
			continue;

		std::vector<unsigned char> pixelsColor(c_pointImageSize * c_pointImageSize * 3, 0);
		std::vector<unsigned char> pixelsBW(c_pointImageSize * c_pointImageSize, 255);

		std::vector<unsigned char> pixelsColorGauss(c_pointImageGaussSize * c_pointImageGaussSize * 3, 0);
		std::vector<unsigned char> pixelsBWGauss(c_pointImageGaussSize * c_pointImageGaussSize, 255);

		for (size_t index = 0; index < points.size(); ++index)
		{
			// Draw the points which are part of this class subset. "i" is a bitmask.
			if ((i & (1 << points[index].classIndex)) != 0)
			{
				int x = (int)Clamp((points[index].p.x * 0.5f + 0.5f) * float(c_pointImageSize - 1), 0.0f, float(c_pointImageSize - 1));
				int y = (int)Clamp((points[index].p.y * 0.5f + 0.5f) * float(c_pointImageSize - 1), 0.0f, float(c_pointImageSize - 1));

				pixelsBW[y * c_pointImageSize + x] = 0;

				unsigned char* pixel = &pixelsColor[(y * c_pointImageSize + x) * 3];

				for (int classIndex = 0; classIndex < 3; ++classIndex)
				{
					if (classIndex == points[index].classIndex)
						pixel[classIndex] = 255;
					else if (pixel[classIndex] == 0)
						pixel[classIndex] = 64;
				}

				x = (int)Clamp((points[index].p.x * 0.5f + 0.5f) * float(c_pointImageGaussSize - 1), 0.0f, float(c_pointImageGaussSize - 1));
				y = (int)Clamp((points[index].p.y * 0.5f + 0.5f) * float(c_pointImageGaussSize - 1), 0.0f, float(c_pointImageGaussSize - 1));

				unsigned char pixelColor[3] = { 64, 64, 64 };
				pixelColor[points[index].classIndex] = 255;
				PlotGaussian<3>(pixelsColorGauss, c_pointImageGaussSize, c_pointImageGaussSize, x, y, c_pointImageGaussBlobSigma, pixelColor);

				unsigned char color[] = { 0 };
				PlotGaussian<1>(pixelsBWGauss, c_pointImageGaussSize, c_pointImageGaussSize, x, y, c_pointImageGaussBlobSigma, color);
			}
		}

		// Write color images
		char fileName[1024];
		sprintf_s(fileName, "%s_%i_%i.%c%c%c.color.png", baseFileName, index, total, (i & 1)?'T':'F', (i & 2) ? 'T' : 'F', (i & 4) ? 'T' : 'F');
		stbi_write_png(fileName, c_pointImageSize, c_pointImageSize, 3, pixelsColor.data(), 0);

		sprintf_s(fileName, "%s_%i_%i.%c%c%c.gauss.color.png", baseFileName, index, total, (i & 1) ? 'T' : 'F', (i & 2) ? 'T' : 'F', (i & 4) ? 'T' : 'F');
		stbi_write_png(fileName, c_pointImageGaussSize, c_pointImageGaussSize, 3, pixelsColorGauss.data(), 0);

		// Write black and white image
		sprintf_s(fileName, "%s_%i_%i.%c%c%c.bw.png", baseFileName, index, total, (i & 1) ? 'T' : 'F', (i & 2) ? 'T' : 'F', (i & 4) ? 'T' : 'F');
		stbi_write_png(fileName, c_pointImageSize, c_pointImageSize, 1, pixelsBW.data(), 0);

		sprintf_s(fileName, "%s_%i_%i.%c%c%c.gauss.bw.png", baseFileName, index, total, (i & 1) ? 'T' : 'F', (i & 2) ? 'T' : 'F', (i & 4) ? 'T' : 'F');
		stbi_write_png(fileName, c_pointImageGaussSize, c_pointImageGaussSize, 1, pixelsBWGauss.data(), 0);
	}
}

void SavePointSet(const std::vector<float2>& points, const char* baseFileName, int index, int total)
{
	// Write out points in text
	{
		char fileName[1024];
		sprintf_s(fileName, "%s_%i_%i.txt", baseFileName, index, total);
		FILE* file = nullptr;
		fopen_s(&file, fileName, "wb");

		fprintf(file, "float points[][2] =\n{\n");
		
		for (size_t index = 0; index < points.size(); ++index)
			fprintf(file, "    { %ff, %ff },\n", points[index].x, points[index].y);

		fprintf(file, "};\n");
		
		fclose(file);
	}

	// Draw an image of the points
	{
		std::vector<unsigned char> pixels(c_pointImageSize * c_pointImageSize, 255);
		std::vector<unsigned char> pixelsGauss(c_pointImageGaussSize * c_pointImageGaussSize, 255);

		for (size_t index = 0; index < points.size(); ++index)
		{
			int x = (int)Clamp((points[index].x * 0.5f + 0.5f) * float(c_pointImageSize - 1), 0.0f, float(c_pointImageSize - 1));
			int y = (int)Clamp((points[index].y * 0.5f + 0.5f) * float(c_pointImageSize - 1), 0.0f, float(c_pointImageSize - 1));
			pixels[y * c_pointImageSize + x] = 0;

			unsigned char color[] = { 0 };
			x = (int)Clamp((points[index].x * 0.5f + 0.5f) * float(c_pointImageGaussSize - 1), 0.0f, float(c_pointImageGaussSize - 1));
			y = (int)Clamp((points[index].y * 0.5f + 0.5f) * float(c_pointImageGaussSize - 1), 0.0f, float(c_pointImageGaussSize - 1));
			PlotGaussian<1>(pixelsGauss, c_pointImageGaussSize, c_pointImageGaussSize, x, y, c_pointImageGaussBlobSigma, color);
		}

		char fileName[1024];
		sprintf_s(fileName, "%s_%i_%i.png", baseFileName, index, total);
		stbi_write_png(fileName, c_pointImageSize, c_pointImageSize, 1, pixels.data(), 0);

		sprintf_s(fileName, "%s_%i_%i.gauss.png", baseFileName, index, total);
		stbi_write_png(fileName, c_pointImageGaussSize, c_pointImageGaussSize, 1, pixelsGauss.data(), 0);
	}
}

void* DummyBatchBegin(const float2& direction)
{
	return nullptr;
}

void DummyBatchEnd(void* param)
{
}

float2 MakeDirection_Gauss(int iterationIndex, int batchIndex, int batchSize)
{
	std::mt19937 rng = GetRNG(iterationIndex * batchSize + batchIndex);
	std::normal_distribution<float> distNormal(0.0f, 1.0f);

	// Make a uniform random unit vector by generating 2 normal distributed values and normalizing the result.
	float2 direction;
	direction.x = distNormal(rng);
	direction.y = distNormal(rng);
	return Normalize(direction);
}

float2 MakeDirection_GoldenRatio(int iterationIndex, int batchIndex, int batchSize)
{
	std::mt19937 rng = GetRNG(batchIndex);
	std::uniform_real_distribution<float> distUniform(0.0f, 1.0f);

	float value01 = distUniform(rng);
	for (int i = 0; i < iterationIndex; ++i)
		value01 = Fract(value01 + c_goldenRatioConjugate);

	float angle = value01 * 2.0f * c_pi;

	return float2
	{
		std::cos(angle),
		std::sin(angle)
	};
}

template <typename TMakeDirectionLambda, typename TBatchBeginLambda, typename TBatchEndLambda, typename TICDFLambda>
void GenerateMulticlassPoints(int numPoints, int weightA, int weightB, int weightC, int numIterations, int batchSize, const char* baseFileName, int numProgressImages, const TMakeDirectionLambda& MakeDirectionLambda, const TBatchBeginLambda& BatchBeginLambda, const TBatchEndLambda& BatchEndLambda, const TICDFLambda& ICDFLambda)
{
	// get the timestamp of when this started
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	printf("==================================\n%s\n==================================\n", baseFileName);

	FILE* file = nullptr;
	char outputFileNameCSV[1024];
	sprintf(outputFileNameCSV, "%s.csv", baseFileName);
	fopen_s(&file, outputFileNameCSV, "wb");
	fprintf(file, "\"Iteration\",\"Avg. Movement\"\n");

	// Generate the starting points
	std::vector<MultiClassPoint> points(numPoints);
	{
		std::mt19937 rng = GetRNG(0);
		std::uniform_real_distribution<float> distUniform(-1.0f, 1.0f);
		for (MultiClassPoint& p : points)
		{
			p.p.x = distUniform(rng);
			p.p.y = distUniform(rng);
		}

		for (int i = 0; i < numPoints; ++i)
		{
			int weightIndex = i % (weightA + weightB + weightC);
			if (weightIndex < weightA)
				points[i].classIndex = 0;
			else if (weightIndex < (weightA + weightB))
				points[i].classIndex = 1;
			else
				points[i].classIndex = 2;
		}
	}

	// Per batch data
	// Each batch entry has it's own data so the batches can be parallelized
	struct BatchData
	{
		BatchData(int numPoints)
		{
			sorted.resize(numPoints);
			for (int i = 0; i < numPoints; ++i)
				sorted[i] = i;
			projections.resize(numPoints);
			batchDirections.resize(numPoints);
		}

		std::vector<int> sorted;
		std::vector<float> projections;
		std::vector<float2> batchDirections;
	};
	std::vector<BatchData> allBatchData(batchSize, BatchData(numPoints));

	// For each iteration
	int lastPercent = -1;
	for (int iterationIndex = 0; iterationIndex < numIterations; ++iterationIndex)
	{
		// Write out progress
		if (numProgressImages > 0)
		{
			int progressInterval = numIterations / numProgressImages;
			if (iterationIndex % progressInterval == 0)
				SavePointSet(points, baseFileName, iterationIndex / progressInterval, numProgressImages);
		}

		// periodically ensure that our classes are well interleaved
		bool interleaveClasses = (iterationIndex % 2) == 0;

		// Do the batches in parallel
		#if MULTITHREADED()
		#pragma omp parallel for
		#endif
		for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
		{
			BatchData& batchData = allBatchData[batchIndex];

			float2 direction = MakeDirectionLambda(iterationIndex, batchIndex, batchSize);

			// project the points
			for (size_t i = 0; i < numPoints; ++i)
				batchData.projections[i] = Dot(direction, points[i].p);

			// sort the projections
			std::sort(batchData.sorted.begin(), batchData.sorted.end(),
				[&](uint32_t a, uint32_t b)
				{
					return batchData.projections[a] < batchData.projections[b];
				}
			);

			// interleave the classes if we should
			if (interleaveClasses)
			{
				// gather the points of each class
				std::vector<int> classesSorted[3];
				for (int i = 0; i < batchData.sorted.size(); ++i)
				{
					int pointIndex = batchData.sorted[i];
					classesSorted[points[pointIndex].classIndex].push_back(pointIndex);
				}

				int classesSortedIndex[3] = { 0, 0, 0 };

				// interleave the sorted classes
				for (int i = 0; i < batchData.sorted.size(); ++i)
				{
					int weightIndex = i % (weightA + weightB + weightC);
					int desiredClassIndex = 0;
					if (weightIndex < weightA)
						desiredClassIndex = 0;
					else if (weightIndex < (weightA + weightB))
						desiredClassIndex = 1;
					else
						desiredClassIndex = 2;

					batchData.sorted[i] = classesSorted[desiredClassIndex][classesSortedIndex[desiredClassIndex]];
					classesSortedIndex[desiredClassIndex]++;
				}
			}

			// update batchDirections
			void* param = BatchBeginLambda(direction);
			for (size_t i = 0; i < numPoints; ++i)
			{
				float targetProjection = ((float(i) + 0.5f) / float(numPoints));

				targetProjection = ICDFLambda(param, targetProjection, direction);

				float projDiff = targetProjection - batchData.projections[batchData.sorted[i]];

				batchData.batchDirections[batchData.sorted[i]].x = direction.x * projDiff;
				batchData.batchDirections[batchData.sorted[i]].y = direction.y * projDiff;
			}
			BatchEndLambda(param);
		}

		// average all batch directions into batchDirections[0]
		{
			for (int batchIndex = 1; batchIndex < batchSize; ++batchIndex)
			{
				float alpha = 1.0f / float(batchIndex + 1);
				for (size_t i = 0; i < numPoints; ++i)
				{
					allBatchData[0].batchDirections[i].x = Lerp(allBatchData[0].batchDirections[i].x, allBatchData[batchIndex].batchDirections[i].x, alpha);
					allBatchData[0].batchDirections[i].y = Lerp(allBatchData[0].batchDirections[i].y, allBatchData[batchIndex].batchDirections[i].y, alpha);
				}
			}
		}

		// update points
		float totalDistance = 0.0f;
		for (size_t i = 0; i < numPoints; ++i)
		{
			const float2& adjust = allBatchData[0].batchDirections[i];

			points[i].p.x += adjust.x;
			points[i].p.y += adjust.y;

			totalDistance += std::sqrt(adjust.x * adjust.x + adjust.y * adjust.y);
		}

		int percent = int(100.0f * float(iterationIndex) / float(numIterations - 1));
		if (percent != lastPercent)
		{
			lastPercent = percent;
			printf("\r[%i%%] %f", percent, totalDistance / float(numPoints));
			fprintf(file, "\"%i\",\"%f\"\n", iterationIndex, totalDistance / float(numPoints));
		}
	}
	printf("\n");

	fclose(file);

	// Write out the final results
	SavePointSet(points, baseFileName, numProgressImages, numProgressImages);

	// report how long this took
	float elpasedSeconds = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - start).count();
	printf("%0.2f seconds\n\n", elpasedSeconds);
}

template <typename TMakeDirectionLambda, typename TBatchBeginLambda, typename TBatchEndLambda, typename TICDFLambda>
void GeneratePoints(int numPoints, int numIterations, int batchSize, const char* baseFileName, int numProgressImages, const TMakeDirectionLambda& MakeDirectionLambda, const TBatchBeginLambda& BatchBeginLambda, const TBatchEndLambda& BatchEndLambda, const TICDFLambda& ICDFLambda)
{
	// get the timestamp of when this started
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	printf("==================================\n%s\n==================================\n", baseFileName);

	FILE* file = nullptr;
	char outputFileNameCSV[1024];
	sprintf(outputFileNameCSV, "%s.csv", baseFileName);
	fopen_s(&file, outputFileNameCSV, "wb");
	fprintf(file, "\"Iteration\",\"Avg. Movement\"\n");

	// Generate the starting points
	std::vector<float2> points(numPoints);
	{
		std::mt19937 rng = GetRNG(0);
		std::uniform_real_distribution<float> distUniform(-1.0f, 1.0f);
		for (float2& p : points)
		{
			p.x = distUniform(rng);
			p.y = distUniform(rng);
		}
	}

	// Per batch data
	// Each batch entry has it's own data so the batches can be parallelized
	struct BatchData
	{
		BatchData(int numPoints)
		{
			sorted.resize(numPoints);
			for (int i = 0; i < numPoints; ++i)
				sorted[i] = i;
			projections.resize(numPoints);
			batchDirections.resize(numPoints);
		}

		std::vector<int> sorted;
		std::vector<float> projections;
		std::vector<float2> batchDirections;
	};
	std::vector<BatchData> allBatchData(batchSize, BatchData(numPoints));

	// For each iteration
	int lastPercent = -1;
	for (int iterationIndex = 0; iterationIndex < numIterations; ++iterationIndex)
	{
		// Write out progress
		if (numProgressImages > 0)
		{
			int progressInterval = numIterations / numProgressImages;
			if (iterationIndex % progressInterval == 0)
				SavePointSet(points, baseFileName, iterationIndex / progressInterval, numProgressImages);
		}

		// Do the batches in parallel
		#if MULTITHREADED()
		#pragma omp parallel for
		#endif
		for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
		{
			BatchData& batchData = allBatchData[batchIndex];

			float2 direction = MakeDirectionLambda(iterationIndex, batchIndex, batchSize);

			// project the points
			for (size_t i = 0; i < numPoints; ++i)
				batchData.projections[i] = Dot(direction, points[i]);

			// sort the projections
			std::sort(batchData.sorted.begin(), batchData.sorted.end(),
				[&](uint32_t a, uint32_t b)
				{
					return batchData.projections[a] < batchData.projections[b];
				}
			);

			// update batchDirections
			void* param = BatchBeginLambda(direction);
			for (size_t i = 0; i < numPoints; ++i)
			{
				float targetProjection = ((float(i) + 0.5f) / float(numPoints));

				targetProjection = ICDFLambda(param, targetProjection, direction);

				float projDiff = targetProjection - batchData.projections[batchData.sorted[i]];

				batchData.batchDirections[batchData.sorted[i]].x = direction.x * projDiff;
				batchData.batchDirections[batchData.sorted[i]].y = direction.y * projDiff;
			}
			BatchEndLambda(param);
		}

		// average all batch directions into batchDirections[0]
		{
			for (int batchIndex = 1; batchIndex < batchSize; ++batchIndex)
			{
				float alpha = 1.0f / float(batchIndex + 1);
				for (size_t i = 0; i < numPoints; ++i)
				{
					allBatchData[0].batchDirections[i].x = Lerp(allBatchData[0].batchDirections[i].x, allBatchData[batchIndex].batchDirections[i].x, alpha);
					allBatchData[0].batchDirections[i].y = Lerp(allBatchData[0].batchDirections[i].y, allBatchData[batchIndex].batchDirections[i].y, alpha);
				}
			}
		}

		// update points
		float totalDistance = 0.0f;
		for (size_t i = 0; i < numPoints; ++i)
		{
			const float2& adjust = allBatchData[0].batchDirections[i];

			points[i].x += adjust.x;
			points[i].y += adjust.y;

			totalDistance += std::sqrt(adjust.x * adjust.x + adjust.y * adjust.y);
		}

		int percent = int(100.0f * float(iterationIndex) / float(numIterations - 1));
		if (percent != lastPercent)
		{
			lastPercent = percent;
			printf("\r[%i%%] %f", percent, totalDistance / float(numPoints));
			fprintf(file, "\"%i\",\"%f\"\n", iterationIndex, totalDistance / float(numPoints));
		}
	}
	printf("\n");

	fclose(file);

	// Write out the final results
	SavePointSet(points, baseFileName, numProgressImages, numProgressImages);

	// report how long this took
	float elpasedSeconds = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - start).count();
	printf("%0.2f seconds\n\n", elpasedSeconds);
}

void MitchellsBestCandidate(int numPoints, const char* baseFileName)
{
	// get the timestamp of when this started
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	printf("==================================\n%s\n==================================\n", baseFileName);

	std::mt19937 rng = GetRNG(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	std::vector<float2> points(numPoints);

	int lastPercent = -1;
	for (int pointIndex = 0; pointIndex < numPoints; ++pointIndex)
	{
		int numCandidates = pointIndex + 1;

		float2 bestCandidate = float2{ 0.0f, 0.0f };
		float bestCandidateScore = -FLT_MAX;

		for (int candidateIndex = 0; candidateIndex < numCandidates; ++candidateIndex)
		{
			float2 candidate = float2{ dist(rng), dist(rng) };

			float minDistance = FLT_MAX;
			for (int testIndex = 0; testIndex < pointIndex; ++testIndex)
				minDistance = std::min(minDistance, DistanceWrap(candidate, points[testIndex]));

			if (minDistance > bestCandidateScore)
			{
				bestCandidateScore = minDistance;
				bestCandidate = candidate;
			}
		}

		points[pointIndex] = bestCandidate;

		int percent = int(100.0f * float(pointIndex) / float(numPoints - 1));
		if (percent != lastPercent)
		{
			lastPercent = percent;
			printf("\r[%i%%]", percent);
		}
	}
	printf("\n");

	// remap all the points from [0,1] to [-1,1]
	for (float2& p : points)
	{
		p.x = p.x * 2.0f - 1.0f;
		p.y = p.y * 2.0f - 1.0f;
	}

	// Write out the final results
	SavePointSet(points, baseFileName, 1, 1);

	// report how long this took
	float elpasedSeconds = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - start).count();
	printf("%0.2f seconds\n\n", elpasedSeconds);
}

void DartThrowing(int numPoints, float minRadius, int maxThrowsPerPoint, const char* baseFileName)
{
	// get the timestamp of when this started
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	printf("==================================\n%s\n==================================\n", baseFileName);

	std::mt19937 rng = GetRNG(0);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	std::vector<float2> points(numPoints);

	float minRadiusSquared = minRadius * minRadius;

	int lastPercent = -1;
	for (int pointIndex = 0; pointIndex < numPoints; ++pointIndex)
	{
		// Throw darts
		bool found = false;
		for (int throwIndex = 0; throwIndex < maxThrowsPerPoint; ++throwIndex)
		{
			float2 newPoint = float2{ dist(rng), dist(rng) };

			bool tooClose = false;
			for (int testIndex = 0; testIndex < pointIndex; ++testIndex)
			{
				if (DistanceWrap(newPoint, points[testIndex]) < minRadius)
				{
					tooClose = true;
					break;
				}
			}

			if (!tooClose)
			{
				found = true;
				points[pointIndex] = newPoint;
				break;
			}
		}

		if (!found)
		{
			printf("[ERROR] DartThrowing ran out of throw attempts at point index %i!\n", pointIndex);
			return;
		}

		int percent = int(100.0f * float(pointIndex) / float(numPoints - 1));
		if (percent != lastPercent)
		{
			lastPercent = percent;
			printf("\r[%i%%]", percent);
		}
	}

	printf("\n");

	// remap all the points from [0,1] to [-1,1]
	for (float2& p : points)
	{
		p.x = p.x * 2.0f - 1.0f;
		p.y = p.y * 2.0f - 1.0f;
	}

	// Write out the final results
	SavePointSet(points, baseFileName, 1, 1);

	// report how long this took
	float elpasedSeconds = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - start).count();
	printf("%0.2f seconds\n\n", elpasedSeconds);
}

int main(int argc, char** argv)
{
	_mkdir("out");

	// Points in small square
	{
		GeneratePoints(1000, 100 * c_sampleMultiplier, 64, "out/square_small", 5, MakeDirection_Gauss, DummyBatchBegin, DummyBatchEnd,
			[](void* param, float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = Square::InverseCDF(y, direction);

				// The CDF is in [-0.5, 0.5], leave the points at half size to show if any left the square
				return x;
			}
		);
	}

	// Points in square
	{
		GeneratePoints(1000, 100 * c_sampleMultiplier, 64, "out/square", 5, MakeDirection_Gauss, DummyBatchBegin, DummyBatchEnd,
			[](void* param, float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = Square::InverseCDF(y, direction);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);
	}

	// Mitchell's best candidate blue noise
	MitchellsBestCandidate(1000, "out/MBC");

	// Dart throwing blue noise (poisson disk)
	DartThrowing(1000, 22.0f / 1000.0f, 100, "out/Dart");

	// Points in circle - naive
	{
		GeneratePoints(1000, 100 * c_sampleMultiplier, 64, "out/circle_naive", 5, MakeDirection_Gauss, DummyBatchBegin, DummyBatchEnd,
			[&](void* param, float y, const float2& direction)
			{
				return y * 2.0f - 1.0f;
			}
		);
	}

	// Multiclass points in a square
	{
		GenerateMulticlassPoints(1000, 1, 4, 16, 100 * c_sampleMultiplier, 64, "out/multiclass_square", 5, MakeDirection_Gauss, DummyBatchBegin, DummyBatchEnd,
			[](void* param, float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = Square::InverseCDF(y, direction);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);
	}

	// Multiclass points in a square, with a density map
	{
		DensityMap densityMap = LoadDensityMap("flower.png");

		GenerateMulticlassPoints(20000, 1, 4, 16, 100 * c_sampleMultiplier, 64, "out/multiclass_square_flower", 5, MakeDirection_Gauss,
			// Batch Begin
			[&] (const float2& direction)
			{
				// Make ICDF by projecting density map onto the direction
				CDF* ret = new CDF;
				*ret = CDFFromDensityMap(densityMap, 1000, direction);
				for (float2& p : ret->CDFSamples)
					p.y -= 0.5f;
				return ret;
			},
			// Batch End
			[] (void* param)
			{
				CDF* cdf = (CDF*)param;
				delete cdf;
			},
			// ICDF
			[](void* param, float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = ((CDF*)param)->InverseCDF(y);


				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);

		// same, with fewer points
		GenerateMulticlassPoints(1000, 1, 4, 16, 100 * c_sampleMultiplier, 64, "out/multiclass_square_flower_fewerPoints", 5, MakeDirection_Gauss,
			// Batch Begin
			[&] (const float2& direction)
			{
				// Make ICDF by projecting density map onto the direction
				CDF* ret = new CDF;
				*ret = CDFFromDensityMap(densityMap, 1000, direction);
				for (float2& p : ret->CDFSamples)
					p.y -= 0.5f;
				return ret;
			},
			// Batch End
			[] (void* param)
			{
				CDF* cdf = (CDF*)param;
				delete cdf;
			},
			// ICDF
			[](void* param, float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = ((CDF*)param)->InverseCDF(y);


				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);
	}

	// Points in square, with a density map
	{
		DensityMap densityMap = LoadDensityMap("flower.png");

		GeneratePoints(20000, 100 * c_sampleMultiplier, 64, "out/square_flower", 5, MakeDirection_Gauss,
			// Batch Begin
			[&] (const float2& direction)
			{
				// Make ICDF by projecting density map onto the direction
				CDF* ret = new CDF;
				*ret = CDFFromDensityMap(densityMap, 1000, direction);
				for (float2& p : ret->CDFSamples)
					p.y -= 0.5f;
				return ret;
			},
			// Batch End
			[] (void* param)
			{
				CDF* cdf = (CDF*)param;
				delete cdf;
			},
			// ICDF
			[](void* param, float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = ((CDF*)param)->InverseCDF(y);


				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);
	}

	// Points in square - batch sizes 1,4,16, 64, 256
	{
		GeneratePoints(1000, 6400 * c_sampleMultiplier, 1, "out/batch1_square", 5, MakeDirection_Gauss, DummyBatchBegin, DummyBatchEnd,
			[](void* param, float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = Square::InverseCDF(y, direction);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);

		GeneratePoints(1000, 1600 * c_sampleMultiplier, 4, "out/batch4_square", 5, MakeDirection_Gauss, DummyBatchBegin, DummyBatchEnd,
			[](void* param, float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = Square::InverseCDF(y, direction);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);

		GeneratePoints(1000, 400 * c_sampleMultiplier, 16, "out/batch16_square", 5, MakeDirection_Gauss, DummyBatchBegin, DummyBatchEnd,
			[](void* param, float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = Square::InverseCDF(y, direction);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);


		GeneratePoints(1000, 100 * c_sampleMultiplier, 64, "out/batch64_square", 5, MakeDirection_Gauss, DummyBatchBegin, DummyBatchEnd,
			[](void* param, float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = Square::InverseCDF(y, direction);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);

		GeneratePoints(1000, 25 * c_sampleMultiplier, 256, "out/batch256_square", 5, MakeDirection_Gauss, DummyBatchBegin, DummyBatchEnd,
			[](void* param, float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = Square::InverseCDF(y, direction);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);
	}

	// Points in square, using golden ratio sequence for random angles
	{
		GeneratePoints(1000, 100 * c_sampleMultiplier, 64, "out/square_GR", 5, MakeDirection_GoldenRatio, DummyBatchBegin, DummyBatchEnd,
			[](void* param, float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = Square::InverseCDF(y, direction);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);
	}

	// Points in circle  - batch size 1
	{
		// make the Numerical ICDF
		CDF circleCDF = CDFFromCDFFn(-0.5f, 0.5f, 1000, UnitCircleCDF);

		GeneratePoints(1000, 6400 * c_sampleMultiplier, 1, "out/batch1_circle", 5, MakeDirection_Gauss, DummyBatchBegin, DummyBatchEnd,
			[&](void* param, float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = circleCDF.InverseCDF(y);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);
	}

	// Points in circle, using golden ratio sequence for random angles
	{
		// make the Numerical ICDF
		CDF circleCDF = CDFFromCDFFn(-0.5f, 0.5f, 1000, UnitCircleCDF);

		GeneratePoints(1000, 100 * c_sampleMultiplier, 64, "out/circle_GR", 5, MakeDirection_GoldenRatio, DummyBatchBegin, DummyBatchEnd,
			[&](void* param, float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = circleCDF.InverseCDF(y);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);
	}

	// Points in circle  - batch size 64
	{
		// make the Numerical ICDF
		CDF circleCDF = CDFFromCDFFn(-0.5f, 0.5f, 1000, UnitCircleCDF);

		GeneratePoints(1000, 100 * c_sampleMultiplier, 64, "out/circle", 5, MakeDirection_Gauss, DummyBatchBegin, DummyBatchEnd,
			[&](void* param, float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = circleCDF.InverseCDF(y);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);
	}

	return 0;
}

/*
Blog Post:
* show a gif of the full 100 steps making noise? we could randomly color the points, so you can follow points by color
* put thing about eyes and the heuristic from nature about hexagon tiles in the middle where it's dense / high quality, and blue noise around the outside where it's sparse / lower quality.
* mention that you could probably do density maps with dart throwing and MBC. Get the density at each point, turn those into distances, and subtract that from the distance between points
? maybe show animations that go from start to end position, over X frames and make them into a gif.
 * That will be a straight line and is the optimal transport. The other animations are the evolution of the OT solve.

 * future: https://dl.acm.org/doi/pdf/10.1145/3550454.3555484

Next: a blog post on deriving that square inverse CDF
Next: figure out how to use sliced OT to make noise masks


DONE:
* mention radon transform and homeography, and link to xrays
* points in circle
 * mention how you can add a z component to make a normalized vector and that it will then be a cosine weighted hemispherical point
* points in square
 * yes, overconvergence does look like a real problem, w/ points in square
 * DFT
 * MBC looks to be higher quality! But, i don't think it can do varying density like sliced OT can.
 * also compare vs dart throwing (cook 86 "Stochastic sampling in computer graphics")
  * average movement graphs?
 * graph of 1,4,16,64,256 batch sizes?
 * then mixed density - the flower density thing
* explain and show multiclass?
* mention other methods to make blue noise exist. like gaussian blue noise.
* not sure why they say they beat dart throwing, when dart throwing looks better to me.


* link to sliced optimal transport sampling. Also the more advanced one? (which is...??)
 * sliced OT sampling http://www.geometry.caltech.edu/pubs/PBCIW+20.pdf
 * more advanced: https://dl.acm.org/doi/pdf/10.1145/3550454.3555484


Not Doing
square:
 * compare using golden ratio for directions, vs random white noise directions
 * note that it's possible to get points outside of the square. up to you if you want to wrap or clamp.  I don't do either, but when drawing the points on the images, i clamp.
 * showing tiling
* the way I did circle ICDF is different than what the sliced OT sampling paper does. They have a numerical ICDF in the end like me, they made with gradient descent. I make a large table with linear interpolation. Seems to work!
 * mention you could make a CDF from a PDF, if it's hard to make the CDF.
* could play around with batch size and see if there are trade offs, and if overconvergence becomes a problem at 1
* note that to get a clearer picture of DFTs, you can make multiple and average them to get the expected DFT which is a lot cleaner and less noisy

*/

