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

void SavePointSet(const std::vector<float>& points, const char* baseFileName, int index, int total)
{
	// Write out points in text
	{
		char fileName[1024];
		sprintf_s(fileName, "%s_%i_%i.txt", baseFileName, index, total);
		FILE* file = nullptr;
		fopen_s(&file, fileName, "wb");

		fprintf(file, "float points[] =\n{\n");
		
		for (size_t index = 0; index < points.size() / 2; ++index)
			fprintf(file, "    %ff, %ff,\n", points[index * 2 + 0], points[index * 2 + 1]);

		fprintf(file, "};\n");
		
		fclose(file);
	}

	// Draw an image of the points
	{
		std::vector<unsigned char> pixels(c_pointImageSize * c_pointImageSize, 255);

		for (size_t index = 0; index < points.size() / 2; ++index)
		{
			int x = (int)Clamp((points[index * 2 + 0] * 0.5f + 0.5f) * float(c_pointImageSize - 1), 0.0f, float(c_pointImageSize - 1));
			int y = (int)Clamp((points[index * 2 + 1] * 0.5f + 0.5f) * float(c_pointImageSize - 1), 0.0f, float(c_pointImageSize - 1));
			pixels[y * c_pointImageSize + x] = 0;
		}

		char fileName[1024];
		sprintf_s(fileName, "%s_%i_%i.png", baseFileName, index, total);
		stbi_write_png(fileName, c_pointImageSize, c_pointImageSize, 1, pixels.data(), 0);
	}
}

void GeneratePointsInCircle(int numPoints, int numIterations, int batchSize, const char* baseFileName, int numProgressImages)
{
	// get the timestamp of when this started
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	printf("==================================\n" __FUNCTION__ " %s\n==================================\n", baseFileName);

	FILE* file = nullptr;
	char outputFileNameCSV[1024];
	sprintf(outputFileNameCSV, "%s.csv", baseFileName);
	fopen_s(&file, outputFileNameCSV, "wb");
	fprintf(file, "\"Iteration\",\"Avg. Movement\"\n");

	// make the Numerical ICDF
	ICDF circleICDF = ICDFFromCDF(-0.5f, 0.5f, 1000, UnitCircleCDF);

	// Generate the starting points
	std::vector<float> points(numPoints * 2);
	{
		std::mt19937 rng = GetRNG(0);
		std::uniform_real_distribution<float> distUniform(-1.0f, 1.0f);
		for (float& f : points)
			f = distUniform(rng);
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
			batchDirections.resize(numPoints * 2);
		}

		std::vector<int> sorted;
		std::vector<float> projections;
		std::vector<float> batchDirections;
	};
	std::vector<BatchData> allBatchData(batchSize, BatchData(numPoints));

	// For each iteration
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
			float direction[2];
			direction[0] = distNormal(rng);
			direction[1] = distNormal(rng);
			float length = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1]);
			direction[0] /= length;
			direction[1] /= length;

			// project the points
			for (size_t i = 0; i < numPoints; ++i)
			{
				batchData.projections[i] =
					direction[0] * points[i * 2 + 0] +
					direction[1] * points[i * 2 + 1];
			}

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
				// Note: circle is from -0.5 to +0.5
				float targetProjection = ((float(i) + 0.5f) / float(numPoints)) - 0.5f;
				targetProjection = circleICDF.InverseCDF(targetProjection);

				// Uncomment this if you want to map it back to [-1,+1]
				targetProjection = targetProjection * 2.0f;

				float projDiff = targetProjection - batchData.projections[batchData.sorted[i]];

				batchData.batchDirections[batchData.sorted[i] * 2 + 0] = direction[0] * projDiff;
				batchData.batchDirections[batchData.sorted[i] * 2 + 1] = direction[1] * projDiff;
			}
		}

		// average all batch directions into batchDirections[0]
		{
			for (int batchIndex = 1; batchIndex < batchSize; ++batchIndex)
			{
				float alpha = 1.0f / float(batchIndex + 1);
				for (size_t i = 0; i < numPoints * 2; ++i)
					allBatchData[0].batchDirections[i] = Lerp(allBatchData[0].batchDirections[i], allBatchData[batchIndex].batchDirections[i], alpha);
			}
		}

		// update points
		float totalDistance = 0.0f;
		for (size_t i = 0; i < numPoints; ++i)
		{
			float adjust[2] = {
				allBatchData[0].batchDirections[i * 2 + 0],
				allBatchData[0].batchDirections[i * 2 + 1]
			};

			points[i * 2 + 0] += adjust[0];
			points[i * 2 + 1] += adjust[1];

			totalDistance += std::sqrt(adjust[0] * adjust[0] + adjust[1] * adjust[1]);
		}

		printf("[%i] %f\n", iterationIndex, totalDistance / float(numPoints));
		fprintf(file, "\"%i\",\"%f\"\n", iterationIndex, totalDistance / float(numPoints));
	}

	fclose(file);

	// Write out the final results
	SavePointSet(points, baseFileName, numProgressImages, numProgressImages);

	// report how long this took
	float elpasedSeconds = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - start).count();
	printf("\n%0.2f seconds\n\n", elpasedSeconds);
}

void GeneratePointsInSquare(int numPoints, int numIterations, int batchSize, const char* baseFileName, int numProgressImages)
{
	// get the timestamp of when this started
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	printf("==================================\n" __FUNCTION__ " %s\n==================================\n", baseFileName);

	FILE* file = nullptr;
	char outputFileNameCSV[1024];
	sprintf(outputFileNameCSV, "%s.csv", baseFileName);
	fopen_s(&file, outputFileNameCSV, "wb");
	fprintf(file, "\"Iteration\",\"Avg. Movement\"\n");

	// Generate the starting points
	std::vector<float> points(numPoints * 2);
	{
		std::mt19937 rng = GetRNG(0);
		std::uniform_real_distribution<float> distUniform(-1.0f, 1.0f);
		for (float& f : points)
			f = distUniform(rng);
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
			batchDirections.resize(numPoints * 2);
		}

		std::vector<int> sorted;
		std::vector<float> projections;
		std::vector<float> batchDirections;
	};
	std::vector<BatchData> allBatchData(batchSize, BatchData(numPoints));

	// For each iteration
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
			float direction[2];
			direction[0] = distNormal(rng);
			direction[1] = distNormal(rng);
			float length = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1]);
			direction[0] /= length;
			direction[1] /= length;

			// project the points
			for (size_t i = 0; i < numPoints; ++i)
			{
				batchData.projections[i] =
					direction[0] * points[i * 2 + 0] +
					direction[1] * points[i * 2 + 1];
			}

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
				// Note: square is from -0.5 to +0.5
				float targetProjection = ((float(i) + 0.5f) / float(numPoints)) - 0.5f;
				targetProjection = Square::InverseCDF(targetProjection, float2{ direction[0], direction[1] });

				// Uncomment this if you want to map it back to [-1,+1]
				targetProjection = targetProjection * 2.0f;

				float projDiff = targetProjection - batchData.projections[batchData.sorted[i]];

				batchData.batchDirections[batchData.sorted[i] * 2 + 0] = direction[0] * projDiff;
				batchData.batchDirections[batchData.sorted[i] * 2 + 1] = direction[1] * projDiff;
			}
		}

		// average all batch directions into batchDirections[0]
		{
			for (int batchIndex = 1; batchIndex < batchSize; ++batchIndex)
			{
				float alpha = 1.0f / float(batchIndex + 1);
				for (size_t i = 0; i < numPoints * 2; ++i)
					allBatchData[0].batchDirections[i] = Lerp(allBatchData[0].batchDirections[i], allBatchData[batchIndex].batchDirections[i], alpha);
			}
		}

		// update points
		float totalDistance = 0.0f;
		for (size_t i = 0; i < numPoints; ++i)
		{
			float adjust[2] = {
				allBatchData[0].batchDirections[i * 2 + 0],
				allBatchData[0].batchDirections[i * 2 + 1]
			};

			points[i * 2 + 0] += adjust[0];
			points[i * 2 + 1] += adjust[1];

			totalDistance += std::sqrt(adjust[0] * adjust[0] + adjust[1] * adjust[1]);
		}

		printf("[%i] %f\n", iterationIndex, totalDistance / float(numPoints));
		fprintf(file, "\"%i\",\"%f\"\n", iterationIndex, totalDistance / float(numPoints));
	}

	fclose(file);

	// Write out the final results
	SavePointSet(points, baseFileName, numProgressImages, numProgressImages);

	// report how long this took
	float elpasedSeconds = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - start).count();
	printf("\n%0.2f seconds\n\n", elpasedSeconds);
}

int main(int argc, char** argv)
{
	_mkdir("out");

	GeneratePointsInSquare(1000, 100, 64, "out/square", 20);
	GeneratePointsInCircle(1000, 100, 16, "out/circle", 20);

	return 0;
}
/*
TODO: could try making things in new gigi editor to test it, also doing these point sets. leave parser for another time.

*/

// TODO: combine GeneratePointsInSquare into a function that takes an ICDF lambda!
// TODO: fix blue noise points to be -1 to 1 i think?
// TODO: does the BN tile?
// TODO; what's the DFT look like?
// TODO: what to do about points outside the square? clamp? wrap?
// TODO: circle not working well?! need ICDF.
// TODO: density map for both circle and square. numerical ICDF!
// TODO: the make the generated points code have a struct for X,Y? instead of just a single array of floats. or make it be like [][2]?

/*
TODO:
- need to solve the thing about all the points going to the egdes! squaring the projection to push them towards the center wasn't the fix. it made a lot of points go to the center!
- compare vs MBC.
- note that the paper has a different way to do circular CDF than you do.
- look for todos

TODO:
1) in a circle. mention how you can make a z coordinate to make it cosine weighted hemispherical points! is overconvergence a problem?
2) in a circle, using golden ratio + initial RNG. see if it converges faster? could also compare stratified points vs pure white noise.
2.5) maybe try all 4 combos: (uniform | stratified) x (uniform | golden ratio)
3) in a square. how to?
4) using a black and white image as a density guide for the dots

NOTES:


Blog Post:
* show derivation of square CDF? and circle if we make it.
* show a gif of the full 100 steps making noise?
* link to sliced optimal transport sampling. Also the more advanced one? (which is...??)
 * sliced OT sampling http://www.geometry.caltech.edu/pubs/PBCIW+20.pdf
 * more advanced: https://dl.acm.org/doi/pdf/10.1145/3550454.3555484
* sliced OT also does multiclass. maybe mention it instead of implementing it? or is it worth implementing?
*/

