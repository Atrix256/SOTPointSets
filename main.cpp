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
				float targetProjection = ((float(i) + 0.5f) / float(numPoints));

				targetProjection = ICDFLambda(targetProjection, float2{ direction[0], direction[1]});

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

	// Points in square
	{
		GeneratePoints(1000, 100, 64, "out/square", 5,
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

	// Points in circle
	{
		// make the Numerical ICDF
		ICDF circleICDF = ICDFFromCDF(-0.5f, 0.5f, 1000, UnitCircleCDF);

		GeneratePoints(1000, 100, 16, "out/circle", 5,
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
* what to do about points outside the square? clamp? wrap?
* density map for both circle and square. numerical ICDF!
* the make the generated points code have a struct for X,Y? instead of just a single array of floats. or make it be like [][2]?
* compare vs MBC.
* is overconvergence a problem?

TODO:
2) in a circle, using golden ratio + initial RNG. see if it converges faster? could also compare stratified points vs pure white noise.
2.5) maybe try all 4 combos: (uniform | stratified) x (uniform | golden ratio)

NOTES:


Blog Post:
* points in circle
 * mention how you can add a z component to make a normalized vector and that it will then be a cosine weighted hemispherical point
* then points in square
 * show the DFT and that it tiles decently!
* then mixed density
* show derivation of square CDF? and circle.
* show a gif of the full 100 steps making noise? we could randomly color the points, so you can follow points by color
* link to sliced optimal transport sampling. Also the more advanced one? (which is...??)
 * sliced OT sampling http://www.geometry.caltech.edu/pubs/PBCIW+20.pdf
 * more advanced: https://dl.acm.org/doi/pdf/10.1145/3550454.3555484
* sliced OT also does multiclass. maybe mention it instead of implementing it? or is it worth implementing?
* the way I did circle ICDF is different than what the sliced OT sampling paper does. They have a numerical ICDF in the end like me, they made with gradient descent. I make a large table with linear interpolation. Seems to work!
 * mention you could make a CDF from a PDF, if it's hard to make the CDF.
Next: figure out how to use sliced OT to make noise masks
*/

