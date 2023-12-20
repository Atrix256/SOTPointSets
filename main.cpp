#define _CRT_SECURE_NO_WARNINGS // for stb

// Settings
#define DETERMINISTIC() true
static const int c_pointImageSize = 256;

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <random>
#include <vector>
#include <direct.h>
#include <stdio.h>
#include <chrono>

#include "squarecdf.h"
#include "NumericalICDF.h"

// TODO: set this back to 10 before making post
static const int c_sampleMultiplier = 1; // 1 for debugging and iteration. 10 for quality

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

		for (size_t index = 0; index < points.size(); ++index)
		{
			int x = (int)Clamp((points[index].x * 0.5f + 0.5f) * float(c_pointImageSize - 1), 0.0f, float(c_pointImageSize - 1));
			int y = (int)Clamp((points[index].y * 0.5f + 0.5f) * float(c_pointImageSize - 1), 0.0f, float(c_pointImageSize - 1));
			pixels[y * c_pointImageSize + x] = 0;
		}

		char fileName[1024];
		sprintf_s(fileName, "%s_%i_%i.png", baseFileName, index, total);
		stbi_write_png(fileName, c_pointImageSize, c_pointImageSize, 1, pixels.data(), 0);
	}
}

template <typename TICDFLambda>
void GeneratePoints(int numPoints, int numIterations, int batchSize, const char* baseFileName, int numProgressImages, const TICDFLambda& ICDFLambda)
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
		#pragma omp parallel for
		for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
		{
			BatchData& batchData = allBatchData[batchIndex];

			std::mt19937 rng = GetRNG(iterationIndex * batchSize + batchIndex);
			std::normal_distribution<float> distNormal(0.0f, 1.0f);

			// Make a uniform random unit vector by generating 2 normal distributed values and normalizing the result.
			float2 direction;
			direction.x = distNormal(rng);
			direction.y = distNormal(rng);
			direction = Normalize(direction);

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
			for (size_t i = 0; i < numPoints; ++i)
			{
				float targetProjection = ((float(i) + 0.5f) / float(numPoints));

				targetProjection = ICDFLambda(targetProjection, direction);

				float projDiff = targetProjection - batchData.projections[batchData.sorted[i]];

				batchData.batchDirections[batchData.sorted[i]].x = direction.x * projDiff;
				batchData.batchDirections[batchData.sorted[i]].y = direction.y * projDiff;
			}
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
				if (DistanceSquared(newPoint, points[testIndex]) < minRadiusSquared)
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

	// Mitchell's best candidate blue noise
	MitchellsBestCandidate(1000, "out/MBC");

	// Dart throwing blue noise (poisson disk)
	DartThrowing(1000, 22.0f / 1000.0f, 100, "out/Dart");

	// Points in square - batch sizes 1,4,16,128
	{
		GeneratePoints(1000, 6400 * c_sampleMultiplier, 1, "out/batch1_square", 5,
			[](float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = Square::InverseCDF(y, direction);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);

		GeneratePoints(1000, 1600 * c_sampleMultiplier, 4, "out/batch4_square", 5,
			[](float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = Square::InverseCDF(y, direction);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);

		GeneratePoints(1000, 400 * c_sampleMultiplier, 16, "out/batch16_square", 5,
			[](float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = Square::InverseCDF(y, direction);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);


		GeneratePoints(1000, 100 * c_sampleMultiplier, 64, "out/batch64_square", 5,
			[](float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = Square::InverseCDF(y, direction);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);

		GeneratePoints(1000, 25 * c_sampleMultiplier, 256, "out/batch256_square", 5,
			[](float y, const float2& direction)
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

	// Points in square - batch size 64
	{
		GeneratePoints(1000, 100 * c_sampleMultiplier, 64, "out/square", 5,
			[] (float y, const float2& direction)
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
		ICDF circleICDF = ICDFFromCDF(-0.5f, 0.5f, 1000, UnitCircleCDF);

		GeneratePoints(1000, 6400 * c_sampleMultiplier, 1, "out/batch1_circle", 5,
			[&](float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = circleICDF.InverseCDF(y);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);
	}

	// Points in circle  - batch size 64
	{
		// make the Numerical ICDF
		ICDF circleICDF = ICDFFromCDF(-0.5f, 0.5f, 1000, UnitCircleCDF);

		GeneratePoints(1000, 100 * c_sampleMultiplier, 16, "out/circle", 5,
			[&](float y, const float2& direction)
			{
				// Convert y: square is in [-0.5, 0.5], but y is in [0, 1].
				y = y - 0.5f;

				// Evaluate ICDF
				float x = circleICDF.InverseCDF(y);

				// The CDF is in [-0.5, 0.5], but we want the points to be in [-1,1]
				return x * 2.0f;
			}
		);
	}

	return 0;
}

/*
TODO:
* density map for both circle and square. numerical ICDF!
 * may need to multiply both CDFs into a table and renormalize. if one is zero where the other isn't, that's lost value.
 * for each pixel, project it to the line. only center point? a pixel might overlap multiple buckets.
* angles in a circle, using golden ratio + initial RNG to see if it converges faster or better?

Blog Post:
* points in circle
 * mention how you can add a z component to make a normalized vector and that it will then be a cosine weighted hemispherical point
 * show average movement graph, maybe compare against golden ratio angles at that point?
* then points in square
 * show the DFT and that it tiles decently! average movement graph too?
 * note that it's possible to get points outside of the square. up to you if you want to wrap or clamp.  I don't do either, but when drawing the points on the images, i clamp.
 * MBC looks to be higher quality! But, i don't think it can do varying density like sliced OT can.
 * also compare vs dart throwing (cook 86 "Stochastic sampling in computer graphics")
 * graph of 1,4,16,64,256 batch sizes?
* using a batch of 1 doesn't look like a good idea (pixels are erratic in batch1_circle and square)
 * over convergence doesn't seem to be a problem, which is nice.
* then mixed density
* show derivation of square CDF? and circle.
* show a gif of the full 100 steps making noise? we could randomly color the points, so you can follow points by color
* link to sliced optimal transport sampling. Also the more advanced one? (which is...??)
 * sliced OT sampling http://www.geometry.caltech.edu/pubs/PBCIW+20.pdf
 * more advanced: https://dl.acm.org/doi/pdf/10.1145/3550454.3555484
* sliced OT also does multiclass. maybe mention it instead of implementing it? or is it worth implementing?
* the way I did circle ICDF is different than what the sliced OT sampling paper does. They have a numerical ICDF in the end like me, they made with gradient descent. I make a large table with linear interpolation. Seems to work!
 * mention you could make a CDF from a PDF, if it's hard to make the CDF.
* could play around with batch size and see if there are trade offs, and if overconvergence becomes a problem at 1
* mention other methods to make blue noise exist. like gaussian blue noise.
Next: figure out how to use sliced OT to make noise masks
*/

