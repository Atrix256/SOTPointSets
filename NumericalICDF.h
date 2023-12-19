#pragma once

#include "maths.h"
#include <vector>
#include <algorithm>

struct ICDF
{
	std::vector<float2> CDFSamples;

	float InverseCDF(float y) const
	{
		auto it = std::lower_bound(CDFSamples.begin(), CDFSamples.end(), y,
			[] (const float2& CDFSample, float y)
			{
				return CDFSample.y < y;
			}
		);

		// If the first item is greater than y, return 0
		if (it == CDFSamples.begin())
		{
			return 0.0f;
		}
		// If the last item is smaller than y, return 1
		else if (it == CDFSamples.end())
		{
			return 1.0f;
		}
		// otherwise, interpolate from it-1 to it
		else
		{
			// The index of the value lower than the value we are searching for
			size_t index = it - CDFSamples.begin() - 1;

			// Get the percent we are between the y values
			float percentY = (y - CDFSamples[index].y) / (CDFSamples[index + 1].y - CDFSamples[index].y);

			// Use that percent to go from the previous to current percent
			return Lerp(CDFSamples[index].x, CDFSamples[index + 1].x, percentY);
		}
	}

	float minX = 0.0f;
	float maxX = 1.0f;
};

template <typename TCDFLambda>
ICDF ICDFFromCDF(float minX, float maxX, int numSamples, const TCDFLambda& CDFLambda)
{
	ICDF ret;
	ret.minX = minX;
	ret.maxX = maxX;

	// Make CDF samples
	ret.CDFSamples.resize(numSamples);
	for (int i = 0; i < numSamples; ++i)
	{
		// We calculate percent this way so we hit 0% and 100% both.
		// Normally you'd want to "keep away from the edges" but we want the full range of data here
		float percent = float(i) / float(numSamples - 1);

		float x = Lerp(minX, maxX, percent);
		float y = CDFLambda(x);
		ret.CDFSamples[i] = float2{ x, y };
	}

	return ret;
}