const apiBaseUrl = "http://localhost:8000";

export interface KernelLabelsResponse {
    id: number;
    weight_coordinate: string;
    labels: string[];
    larger_patterns: string[];
    created_at: string;
    updated_at: string;
}

export interface KernelLabelsRequest {
    weight_coordinate: string;
    labels: string[];
}

export interface AddRemoveLabelResponse {
    success: boolean;
    labels: string[];
}


export interface AddRemoveLargerPatternResponse {
    success: boolean;
    larger_patterns: string[];
}

export async function getKernelLabels(weightCoordinate: string): Promise<KernelLabelsResponse> {
    const response = await fetch(
        `${apiBaseUrl}/api/kernel-labels/${encodeURIComponent(weightCoordinate)}/`,
        {
            method: 'GET',
        }
    );

    if (!response.ok) {
        throw new Error(`Failed to fetch kernel labels: ${response.status}`);
    }

    return response.json();
}

export async function updateKernelLabels(
    weightCoordinate: string, 
    labels: string[]
): Promise<KernelLabelsResponse> {
    const response = await fetch(
        `${apiBaseUrl}/api/kernel-labels/${encodeURIComponent(weightCoordinate)}/`,
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                weight_coordinate: weightCoordinate,
                labels: labels,
            }),
        }
    );

    if (!response.ok) {
        throw new Error(`Failed to update kernel labels: ${response.status}`);
    }

    return response.json();
}

export async function addKernelLabel(
    weightCoordinate: string, 
    label: string
): Promise<AddRemoveLabelResponse> {
    const response = await fetch(
        `${apiBaseUrl}/api/kernel-labels/${encodeURIComponent(weightCoordinate)}/add-label/?label=${encodeURIComponent(label)}`,
        {
            method: 'POST',
        }
    );

    if (!response.ok) {
        throw new Error(`Failed to add kernel label: ${response.status}`);
    }

    return response.json();
}

export async function removeKernelLabel(
    weightCoordinate: string, 
    label: string
): Promise<AddRemoveLabelResponse> {
    const response = await fetch(
        `${apiBaseUrl}/api/kernel-labels/${encodeURIComponent(weightCoordinate)}/remove-label/?label=${encodeURIComponent(label)}`,
        {
            method: 'POST',
        }
    );

    if (!response.ok) {
        throw new Error(`Failed to remove kernel label: ${response.status}`);
    }

    return response.json();
}


export async function addLargerPattern(
    weightCoordinate: string, 
    pattern: string
): Promise<AddRemoveLargerPatternResponse> {
    const response = await fetch(
        `${apiBaseUrl}/api/kernel-labels/${encodeURIComponent(weightCoordinate)}/add-larger-pattern/?pattern=${encodeURIComponent(pattern)}`,
        {
            method: 'POST',
        }
    );

    if (!response.ok) {
        throw new Error(`Failed to add larger pattern: ${response.status}`);
    }

    return response.json();
}

export async function removeLargerPattern(
    weightCoordinate: string, 
    pattern: string
): Promise<AddRemoveLargerPatternResponse> {
    const response = await fetch(
        `${apiBaseUrl}/api/kernel-labels/${encodeURIComponent(weightCoordinate)}/remove-larger-pattern/?pattern=${encodeURIComponent(pattern)}`,
        {
            method: 'POST',
        }
    );

    if (!response.ok) {
        throw new Error(`Failed to remove larger pattern: ${response.status}`);
    }

    return response.json();
}